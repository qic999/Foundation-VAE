# Augmentation

This folder contains the augmentation pipeline and segmentation training setup used in **Foundation-VAE**.  
For segmentation experiments, we use **nnU-Net** on reconstructed/augmented CT data.

## 1. Data

Augmentation-related data and checkpoints are released on Hugging Face:

- **Augmentation release**: [https://huggingface.co/qicq1c/Foundation-VAE/tree/main/Augmentation](https://huggingface.co/qicq1c/Foundation-VAE/tree/main/Augmentation)
- **Project root**: [https://huggingface.co/qicq1c/Foundation-VAE](https://huggingface.co/qicq1c/Foundation-VAE)

Please download the dataset package(s) and place them into your local `nnUNet_raw` structure.

## 2. Environment

### Option A (pip install nnunetv2)
```bash
conda create -n nnunet python=3.10 -y
conda activate nnunet
pip install --upgrade setuptools packaging
pip install nnunetv2
```

### Option B (install from source)
```bash
conda create -n nnunet python=3.10 -y
conda activate nnunet
pip install --upgrade setuptools packaging
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
cd ..
```

## 3. nnU-Net paths
```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

## 4. Data format
Prepare data in standard nnU-Net format:
```
${nnUNet_raw}/Dataset009_FoundationVAE/
├── dataset.json
├── imagesTr
├── labelsTr
├── imagesTs
└── labelsTs   # optional
```
Use nnU-Net naming rules for files (e.g., case_0001_0000.nii.gz for image, case_0001.nii.gz for label).

## 5. Training
```bash
nnUNetv2_plan_and_preprocess -d 9 --verify_dataset_integrity
nnUNetv2_train 9 3d_fullres 0
```

## 6. Inference
```bash
nnUNetv2_predict \
  -i ${nnUNet_raw}/Dataset009_FoundationVAE/imagesTs \
  -o ./predictions \
  -d 9 \
  -c 3d_fullres \
  -f 0
```

## 7. Notes
- We use nnU-Net as the segmentation backbone for augmentation evaluation.
- The key comparison is training with different input views (original CT vs reconstructed/augmented CT).
- For exact splits and released artifacts, refer to the Hugging Face links above.