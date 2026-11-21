
# BreastCare-AI

### Multi-Task Deep Learning for Breast Ultrasound Classification and Lesion Segmentation

BreastCare-AI is a deep learning research prototype designed to analyze breast ultrasound images through two simultaneous tasks:

1. **Image Classification:** Classifies ultrasound images into Benign, Malignant, or Normal categories.
2. **Lesion Segmentation:** Generates a pixel-level predicted lesion mask to highlight potentially relevant image regions.

The project uses a custom multi-task CNN architecture combining a CNN encoder, classification branch, and U-Net-inspired segmentation decoder.

> **Disclaimer:** This is an academic research prototype and is not clinically validated. It must not be used for medical diagnosis or treatment decisions.

---

## 1. Project Objectives

- Develop a multi-task deep learning model for breast ultrasound analysis.
- Perform three-class image classification.
- Generate segmentation masks for lesion localization.
- Investigate classification and segmentation performance together.
- Analyze segmentation false positives and lesion preservation.
- Provide an image inference pipeline with visual outputs.

## 2. Dataset

The project uses the Breast Ultrasound Images Dataset (BUSI).

| Category | Number of Images |
|---|---:|
| Benign | 437 |
| Malignant | 210 |
| Normal | 133 |
| **Total** | **780** |

### Dataset Split

The dataset was divided at image level using a 70:15:15 split with random seed 42.

| Split | Images |
|---|---:|
| Training | 545 |
| Validation | 115 |
| Testing | 120 |
| **Total** | **780** |

Note: The split is image-level, not patient-independent. Therefore, patient-level generalization has not been established.

## 3. Model Architecture

The project implements a custom MultiTaskCNN.

### Encoder

| Layer | Channels |
|---|---:|
| Encoder 1 | 32 |
| Encoder 2 | 64 |
| Encoder 3 | 128 |
| Encoder 4 | 256 |
| Bottleneck | 512 |

### Classification Branch

The classification branch uses:

- Adaptive Average Pooling
- Flatten layer
- Fully Connected layer (512 → 256)
- ReLU activation
- Dropout (0.4)
- Output layer (256 → 3)

### Segmentation Branch

The segmentation decoder uses skip connections from the encoder and progressively upsamples feature maps.

The final segmentation head produces a single-channel lesion probability map.

### Model Specifications

- Architecture: Custom MultiTaskCNN
- Input size: 224 × 224 × 3
- Classification outputs: 3
- Segmentation outputs: 1
- Total parameters: Approximately 7.98 million
- Framework: PyTorch

## 4. Image Preprocessing

The preprocessing pipeline includes:

1. Convert input image to RGB.
2. Resize image to 224 × 224.
3. Apply bilinear interpolation.
4. Normalize pixel values to the range [-1, 1].
5. Convert image to PyTorch tensor.

Segmentation masks are resized using nearest-neighbor interpolation. Multiple masks are combined using a binary union operation.

## 5. Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Batch Size | 2 |
| Random Seed | 42 |
| Selected Checkpoint | Epoch 12 |
| Training Device | CUDA GPU |

The training objective combines classification and segmentation losses:

- Cross-Entropy Loss
- Binary Cross-Entropy with Logits
- Dice Loss

The selected checkpoint is:

`checkpoints/best_multitask_cnn_gpu.pth`

## 6. Evaluation Results

### Classification Performance

The final test-set evaluation produced the following results:

| Metric | Result |
|---|---:|
| Accuracy | 81.67% |
| Correct Predictions | 98 / 120 |
| Incorrect Predictions | 22 / 120 |
| Macro F1-score | 0.8109 |

### Test Confusion Matrix

Rows represent actual classes and columns represent predicted classes.

| Actual / Predicted | Benign | Malignant | Normal |
|---|---:|---:|---:|
| Benign | 57 | 10 | 0 |
| Malignant | 7 | 25 | 0 |
| Normal | 1 | 4 | 16 |

### Segmentation Performance

| Metric | Result |
|---|---:|
| Mean Lesion Dice Score | 0.6828 |
| Lesion Images Evaluated | 99 |
| Normal Images Evaluated | 21 |
| Normal Images with Non-empty Masks | 6 |

The normal-image mask rate represents segmentation false positives, not classification false positives.

### Important Evaluation Note

The test set was evaluated once for baseline reporting. Subsequent threshold and filtering investigations were performed using validation data.

## 7. Segmentation Post-processing

The inference pipeline supports optional connected-component filtering.

Experimental configuration:

| Parameter | Value |
|---|---:|
| Segmentation Threshold | 0.70 |
| Minimum Component Area | 200 pixels |
| Minimum Mean Component Probability | 0.80 |

The original model mask is preserved. Filtering is optional and does not modify the trained checkpoint.

The filtering approach showed improvements in some validation metrics but also reduced lesion overlap in certain images. It is therefore an experimental post-processing option, not a clinically validated improvement.

## 8. Project Structure

```text
BreastCare-AI/
│
├── checkpoints/
│   └── best_multitask_cnn_gpu.pth
│
├── data/
│   ├── raw/
│   │   └── BUSI/
│   │       ├── benign/
│   │       ├── malignant/
│   │       └── normal/
│   │
│   └── processed/
│       └── ultrasound/
│
├── docs/
│   └── research/
│
├── outputs/
│   └── predictions/
│
├── src/
│   ├── data/
│   │   ├── busi_dataset.py
│   │   └── preprocessing.py
│   │
│   ├── models/
│   │   ├── multitask_cnn.py
│   │   └── losses.py
│   │
│   ├── training/
│   │   └── train.py
│   │
│   └── inference.py
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 9. Installation

### Requirements

- Python 3.11
- PyTorch
- CUDA-compatible GPU (optional)
- Required Python packages from requirements.txt

### Setup

Clone the repository:

```bash
git clone https://github.com/dev-aryan24/BreastCare-AI.git
cd BreastCare-AI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 10. Run Inference

Run the model on an ultrasound image:

```powershell
python -m src.inference --image "path/to/image.png"
```

To enable experimental segmentation filtering:

```powershell
python -m src.inference --image "path/to/image.png" --apply-filter
```

The inference pipeline generates:

- Original segmentation mask
- Filtered segmentation mask
- Original segmentation overlay
- Filtered segmentation overlay

Generated files are saved in:

```text
outputs/predictions/
```

## 11. Technologies Used

- Python
- PyTorch
- NumPy
- Pillow
- SciPy
- CUDA
- Matplotlib
- Pandas
- OpenCV

## 12. Research Limitations

- The model has been evaluated on a relatively small dataset.
- The dataset split is not patient-independent.
- Classification errors remain, particularly between benign and malignant classes.
- Segmentation false positives remain on some normal images.
- Segmentation quality varies across individual images.
- External dataset validation has not been established.
- The model has not undergone clinical validation or regulatory assessment.

## 13. Future Scope

Potential future development includes:

- External dataset evaluation.
- Patient-independent validation.
- Improved generalization across ultrasound equipment.
- Better segmentation false-positive handling.
- Model interpretability and explainability.
- Independent clinical evaluation.

## 14. Author

**BreastCare-AI**

An academic deep learning research project focused on breast ultrasound image classification and lesion segmentation.

---

**Research Prototype | Deep Learning | Medical Image Analysis**
