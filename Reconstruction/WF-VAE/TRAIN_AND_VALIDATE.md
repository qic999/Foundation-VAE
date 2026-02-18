# Data Preparation

The training data can be organized efficiently by placing all videos recursively within a single directory. This approach simplifies the process, enabling seamless integration of multiple datasets for training purposes. To implement this, you may [download](https://github.com/cvdfoundation/kinetics-dataset) the Kinetics-400 dataset for both training and testing.

```
Training Dataset
|——sub_dataset1
    |——sub_sub_dataset1
        |——video1.mp4
        |——video2.mp4
        ......
    |——sub_sub_dataset2
        |——video3.mp4
        |——video4.mp4
        ......
|——sub_dataset2
    |——video5.mp4
    |——video6.mp4
    ......
|——video7.mp4
|——video8.mp4
```

# Training

To train the model using your dataset, update the `--video_path` and `--eval_video_path` parameters in `examples/train_ddp.sh` to point to your dataset. Then, execute the script by running:

```
bash examples/train_ddp.sh
```

This command will initiate the training process. Ensure that you are logged into your wandb account before starting the training.

Below, we introduce the key arguments necessary for training:

| Argparse | Usage |
|:---|:---|
|_Training size_||
|`--num_frames`|The number of using frames for training videos|
|`--resolution`|The resolution of the input to the VAE|
|`--batch_size`|The local batch size in each GPU|
|`--sample_rate`|The frame interval of when loading training videos|
|_Data processing_||
|`--video_path`|/path/to/dataset|
|_Load weights_||
|`--model_name`| `CausalVAE` or `WFVAE`|
|`--model_config`|/path/to/config.json The model config of VAE. If you want to train from scratch use this parameter.|
|`--pretrained_model_name_or_path`|A directory containing a model checkpoint and its config. Using this parameter will only load its weight but not load the state of the optimizer|
|`--resume_from_checkpoint`|/path/to/checkpoint.ckpt. It will resume the training process from the checkpoint including the weight and the optimizer.|

# Validation

The evaluation process consists of two steps:

Reconstruct videos in batches: `bash examples/gen_video.sh`
Evaluate video metrics: `bash examples/eval.sh`

To simplify the evaluation, environment variables are used for control. For step 1 (`bash examples/gen_video.sh`):


```bash
# Experiment name
EXP_NAME=test
# Video parameters
SAMPLE_RATE=1
NUM_FRAMES=33
RESOLUTION=256
# Model weights
CKPT=ckpt
# Select subset size (0 for full set)
SUBSET_SIZE=0
# Dataset directory
DATASET_DIR=test_video
```

For step 2 (`bash examples/eval.sh`):

```bash
# Experiment name
EXP_NAME=test
# Video parameters
SAMPLE_RATE=1
NUM_FRAMES=33
RESOLUTION=256
# Evaluation metric
METRIC=lpips
# Select subset size (0 for full set)
SUBSET_SIZE=0
# Path to the ground truth videos, which can be saved during video reconstruction by setting `--output_origin`
REAL_DATASET_DIR=video_gen/${EXP_NAME}_sr${SAMPLE_RATE}_nf${NUM_FRAMES}_res${RESOLUTION}_subset${SUBSET_SIZE}/origin
```