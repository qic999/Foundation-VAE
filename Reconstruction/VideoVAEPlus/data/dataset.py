import os
import random
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import pandas as pd


class DatasetVideoLoader(Dataset):
    """
    Dataset for loading videos and captions from a CSV file.
    CSV file contains two columns: 'path' and 'text', where:
        - 'path' is the path to the video file
        - 'text' is the caption for the video.
    """

    def __init__(
        self,
        csv_file,
        resolution,
        video_length,
        frame_stride=4,
        subset_split="all",
        clip_length=1.0,
        random_stride=False,
        mode="video",
    ):
        self.csv_file = csv_file
        self.resolution = resolution
        self.video_length = video_length
        self.subset_split = subset_split
        self.frame_stride = frame_stride
        self.clip_length = clip_length
        self.random_stride = random_stride
        self.mode = mode

        assert self.subset_split in ["train", "test", "val", "all"]
        self.exts = ["avi", "mp4", "webm"]

        if isinstance(self.resolution, int):
            self.resolution = [self.resolution, self.resolution]

        # Load dataset from CSV file
        self._make_dataset()

    def _make_dataset(self):
        """
        Load video paths and captions from the CSV file.
        """
        self.videos = pd.read_csv(self.csv_file)
        print(f"Loaded {len(self.videos)} videos from {self.csv_file}")

        if self.subset_split == "val":
            self.videos = self.videos[-300:]
        elif self.subset_split == "train":
            self.videos = self.videos[:-300]
        elif self.subset_split == "test":
            self.videos = self.videos[-30:]

        print(f"Number of videos = {len(self.videos)}")

        # Create video indices for image mode
        self.video_indices = list(range(len(self.videos)))

    def set_mode(self, mode):
        self.mode = mode

    def _get_video_path(self, index):
        return self.videos.iloc[index]["path"]

    def __getitem__(self, index):
        if self.mode == "image":
            return self.__getitem__images(index)
        else:
            return self.__getitem__video(index)

    def __getitem__video(self, index):
        while True:
            video_path = self.videos.iloc[index]["path"]
            caption = self.videos.iloc[index]["text"]

            try:
                video_reader = VideoReader(
                    video_path,
                    ctx=cpu(0),
                    width=self.resolution[1],
                    height=self.resolution[0],
                )
                if len(video_reader) < self.video_length:
                    index = (index + 1) % len(self.videos)
                    continue
                else:
                    break
            except Exception as e:
                print(f"Load video failed! path = {video_path}, error: {str(e)}")
                index = (index + 1) % len(self.videos)
                continue

        if self.random_stride:
            self.frame_stride = random.choice([4, 8, 12, 16])

        all_frames = list(range(0, len(video_reader), self.frame_stride))
        if len(all_frames) < self.video_length:
            all_frames = list(range(0, len(video_reader), 1))

        # Select random clip
        rand_idx = random.randint(0, len(all_frames) - self.video_length)
        frame_indices = all_frames[rand_idx : rand_idx + self.video_length]
        frames = video_reader.get_batch(frame_indices)
        assert (
            frames.shape[0] == self.video_length
        ), f"{len(frames)}, self.video_length={self.video_length}"

        frames = (
            torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        )  # [t,h,w,c] -> [c,t,h,w]
        assert (
            frames.shape[2] == self.resolution[0]
            and frames.shape[3] == self.resolution[1]
        ), f"frames={frames.shape}, self.resolution={self.resolution}"
        frames = (frames / 255 - 0.5) * 2

        return {"video": frames, "caption": caption, "is_video": True}

    def __getitem__images(self, index):
        frames_list = []
        for i in range(self.video_length):
            # Get a unique video for each frame
            video_index = (index + i) % len(self.video_indices)
            video_path = self._get_video_path(video_index)

            try:
                video_reader = VideoReader(
                    video_path,
                    ctx=cpu(0),
                    width=self.resolution[1],
                    height=self.resolution[0],
                )
            except Exception as e:
                print(f"Load video failed! path = {video_path}, error = {e}")
                # Skip this video and try the next one
                return self.__getitem__images((index + 1) % len(self.video_indices))

            # Randomly select a frame from the video
            rand_idx = random.randint(0, len(video_reader) - 1)
            frame = video_reader[rand_idx]
            frame_tensor = (
                torch.tensor(frame.asnumpy()).permute(2, 0, 1).float().unsqueeze(0)
            )  # [h,w,c] -> [c,h,w] -> [1, c, h, w]

            frames_list.append(frame_tensor)

        frames = torch.cat(frames_list, dim=0)
        frames = (frames / 255 - 0.5) * 2
        frames = frames.permute(1, 0, 2, 3)
        assert (
            frames.shape[2] == self.resolution[0]
            and frames.shape[3] == self.resolution[1]
        ), f"frame={frames.shape}, self.resolution={self.resolution}"

        data = {"video": frames, "is_video": False}
        return data

    def __len__(self):
        return len(self.videos)
