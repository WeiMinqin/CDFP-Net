# CDFP-Net : Cross-Modal Dynamic Fusion with Diffusion Priors for PET-CT Tumor Segmentation
CDFP-Net is a novel dual-stream multimodal segmentation framework designed to leverage large-scale generative pre-training priors to enhance the accuracy and robustness of PET-CT tumor segmentation.
![](fig/fig1.png)
## Abstract
Accurate automated tumor segmentation in PET-CT imaging is critical for clinical diagnosis and radiotherapy planning. However, limited annotated data, the high computational cost of volumetric modeling, and intrinsic physical discrepancies between metabolic PET and anatomical CT modalities pose substantial challenges. To address these issues, we propose CDFP-Net, a dual-stream framework that integrates generative diffusion priors with cross-modal dynamic graph fusion. Specifically, we employ a frozen, self-supervised pre-trained Denoising Diffusion Probabilistic Model (DDPM) as a robust feature prior, augmented with lightweight adapters to enhance modality-specific representations under limited supervision. To effectively bridge cross-modal discrepancies, we design a modality-asymmetric dynamic graph fusion mechanism in which metabolically active PET regions act as spatial anchors to query and aggregate complementary anatomical boundary cues from CT. To alleviate the computational burden of full 3D processing while preserving global semantic topology, we further introduce a Multi-Angle Maximum Intensity Projection (MA-MIP) strategy for efficient volumetric context modeling. Extensive experiments on three multi-center datasets (AutoPET, HECKTOR, and PCLT20K) demonstrate that CDFP-Net consistently outperforms state-of-the-art methods in both accuracy and robustness, highlighting its strong potential for precise biological target volume (BTV) delineation in clinical practice.
## Requirement
```pip install -r requirement.txt```
## Preprocessing
The data preprocessing workflow of this project (including multi-angle MIP generation and intensity normalization) is referenced from [MIP-DDPM](https://github.com/Amirhosein2c/MIP-DDPM/tree/main/Data_Preparation).
## Pre-training
```python SS_diff.py```

Note: CT and PET should be pre-trained separately.
## Downstream Fine-Tuning
```python Diff_Seg_early.py```
## 🚀 Quick Start: HECKTOR Dataset Case Study
To help you easily reproduce our results and apply CDFP-Net to your own research, we provide a complete pipeline tutorial using the HECKTOR dataset as an example.

### 1. Data Preprocessing
To standardize the raw NIfTI files of the HECKTOR dataset, we adopted the preprocessing pipeline proposed by Cai et al. Please refer to their repository for the initial setup:

Preprocessing Repository: [HECKTOR2025-MEDAI](https://github.com/Liiiii2101/HECKTOR2025-MEDAI)

After completing the NIfTI preprocessing, perform the Multi-Angle Maximum Intensity Projection (MA-MIP). Then, use the scripts provided in the data/ directory to split the dataset for subsequent training and testing.

### 2. Pre-trained Weights for Downstream Training
To accelerate convergence and achieve optimal performance on your downstream segmentation tasks, we provide our self-supervised pre-trained weights for both CT and PET modalities.

You can find the pre-trained weights in the following directories:

CT Pre-trained Model: CDFP/pretrain/CT/CT.pth

PET Pre-trained Model: CDFP/pretrain/PET/PET.pth

To train the downstream model using these weights, run:
```python train.py --dataset HECKTOR --resume_ct CDFP/pretrain/CT/CT.pth --resume_pet CDFP/pretrain/PET/PET.pth --batch_size 4```
### 3. Direct Testing with Downstream Weights
If you wish to skip the training phase and directly evaluate the performance of CDFP-Net on the HECKTOR test set, we also provide the fully fine-tuned downstream weights.

Downstream Checkpoint: CDFP/downstream/checkpoint.pth

To run inference and calculate metrics (e.g., Dice, Hausdorff Distance), execute:

```python test.py --dataset HECKTOR --weights CDFP/downstream/checkpoint.pth --save_predictions True```
## Acknowledgment
Code copied a lot from [GenSelfDiff-HIS](https://github.com/suhas-srinath/GenSelfDiff-HIS/tree/main)、[MIP-DDPM](https://github.com/Amirhosein2c/MIP-DDPM/tree/main/Data_Preparation)、[AAHN](https://github.com/joker-527/AAHN)、[HECKTOR2025-MEDAI](https://github.com/Liiiii2101/HECKTOR2025-MEDAI).

