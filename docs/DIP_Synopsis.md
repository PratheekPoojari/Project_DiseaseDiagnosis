# PROJECT SYNOPSIS
## A MULTIMODAL DEEP LEARNING FRAMEWORK FOR DERMATOLOGICAL DISEASE DIAGNOSIS
### PRIMARY FOCUS: DIGITAL IMAGE PROCESSING & DEEP CONVOLUTIONAL NETWORKS (ConvNeXt-Base)

---

### Academic Identification
- **Project Title:** Multimodal Skin Disease Diagnosis System via Digital Image Processing and Deep Convolutional Neural Networks
- **Course Submission:** Digital Image Processing (DIP) Capstone Project
- **Candidate Name:** Pratheek Poojari
- **Degree / Semester:** Bachelor of Computer Applications (BCA), 5th Semester
- **Institution:** Department of Computer Applications, Surana College, Bangalore
- **Affiliated University:** Bangalore University, Karnataka, India
- **Project Guide:** Mrs. Madhushree B S, Assistant Professor
- **Submission Date:** September 2026

---

## 1. INTRODUCTION

Dermatological diseases represent one of the most widespread healthcare burdens globally, affecting over 1.8 billion individuals at any given time. In India, the clinical diagnosis of cutaneous pathology faces severe systemic constraints: there are fewer than 12,000 certified dermatologists serving a population exceeding 1.4 billion (a specialist-to-patient ratio below 1:116,000). Furthermore, over 80% of these specialists are concentrated in major metropolitan centers, leaving rural and semi-urban districts critically underserved.

Digital Image Processing (DIP) and Computer Vision (CV) offer transformative potential to democratize dermatological triage. Deep learning models applied to clinical photographs can automatically detect subtle morphological features—such as the pearly rolled border of a *Basal Cell Carcinoma*, the pigment variegation of a *Melanoma*, the micaceous silvery scaling of *Psoriasis*, or the annular erythema of a *Fungal Infection*. 

This project develops an enterprise-grade automated skin disease diagnosis system. While the overall platform operates multimodally, this synopsis delineates the **Digital Image Processing (DIP)** branch. The visual architecture is powered by **ConvNeXt-Base** (88 million parameters), trained on 27,153 clinical dermatological images across ten complex conditions, achieving a state-of-the-art **86.92% Validation Accuracy** using distributed multi-GPU training.

---

## 2. PROBLEM STATEMENT

Manual visual inspection of skin lesions is subjective, qualitative, and heavily reliant on specialist experience. Many dermatological conditions present overlapping macroscopic morphological patterns:
1. **Benign vs. Malignant Pigmented Lesions:** Early-stage *Melanoma* frequently mimics benign *Melanocytic Nevi* (common moles) or *Seborrheic Keratoses*, leading to delayed biopsy and catastrophic metastatic progression.
2. **Inflammatory vs. Infectious Eruptions:** Eczematous dermatitis and superficial dermatophytosis (fungal infections) both manifest as erythematous scaling plaques. Treating fungal infections with topical steroids due to visual misclassification induces severe complications (tinea incognito).
3. **Photographic & Environmental Variances:** Clinical images captured via mobile devices suffer from uncontrolled illumination, flash reflections, varying camera sensor noise, and diverse patient skin tones.

An automated DIP system must process raw, noisy patient photographs, apply robust color space transformations and geometric augmentations, and extract high-level feature representations capable of distinguishing visually subtle disease boundaries with high clinical accuracy.

---

## 3. PROJECT OBJECTIVES

1. **DIP Architecture Objective:** Implement, fine-tune, and optimize a state-of-the-art Convolutional Neural Network (**ConvNeXt-Base**, 88M parameters) using transfer learning from ImageNet-1K pretrained weights.
2. **Preprocessing & Augmentation Objective:** Construct an image preprocessing and augmentation pipeline that handles color-space fidelity (BGR to RGB preservation), aspect-ratio preserving cropping, and illumination/rotational invariance.
3. **Distributed Multi-GPU Training Objective:** Execute high-throughput distributed training via PyTorch **DistributedDataParallel (DDP)** on dual NVIDIA Tesla T4 GPUs with Automatic Mixed Precision (AMP FP16).
4. **Regularization & Optimization Objective:** Implement defensive mechanisms against underfitting and overfitting, including Label Smoothing ($\alpha = 0.1$), Decoupled Weight Decay ($\lambda = 0.01$), Cosine Annealing learning rate schedules, and Early Stopping.
5. **Multimodal Consensus Integration:** Integrate the visual classification pipeline with a companion Clinical Natural Language Processing branch (**Bio_ClinicalBERT**) via a **Late Decision Fusion Layer (60% Image / 40% Text)** to resolve diagnostic ambiguity.
6. **Production Deployment:** Deliver a responsive Streamlit web application supporting live mobile camera feeds via USB-C, multi-format clinical exports, and longitudinal health tracking.

---

## 4. DATASET SPECIFICATION & SCOPE

The DIP branch utilizes the comprehensive *Skin Diseases Image Dataset* (ismailpromus), encompassing **27,153 clinical images** partitioned across **ten distinct dermatological classes**:

| # | Class Name | Clinical Category | Image Count | Visual Morphological Hallmarks |
|---|---|---|:---:|---|
| 1 | `Eczema` | Inflammatory | 1,677 | Erythematous, ill-defined patches with weeping papulovesicles and crusting. |
| 2 | `Melanoma` | Malignancy | 15,750 | ABCDE criteria: Asymmetry, irregular borders, color variegation, diameter $>6$mm. |
| 3 | `Atopic Dermatitis` | Inflammatory | 1,250 | Lichenified plaques, severe excoriations concentrated in flexural creases. |
| 4 | `Basal Cell Carcinoma` | Malignancy | 3,323 | Translucent pearly nodule with prominent telangiectasia and rolled borders. |
| 5 | `Melanocytic Nevi` | Benign Neoplasm | 7,970 | Symmetric, uniformly pigmented, circumscribed brown macules or papules. |
| 6 | `Benign Keratosis-like Lesions` | Benign Neoplasm | 2,624 | Solar lentigines and verrucous macules with sharply demarcated borders. |
| 7 | `Psoriasis Lichen Planus` | Autoimmune | 2,000 | Micaceous silvery scales on well-demarcated salmon-pink plaques. |
| 8 | `Seborrheic Keratoses` | Benign Neoplasm | 1,800 | Greasy, waxy, hyperpigmented "stuck-on" verrucous plaques. |
| 9 | `Fungal Infections` | Cutaneous Infection | 1,700 | Annular scaling plaques with advancing borders and central clearing (ringworm). |
| 10| `Viral Infections` | Cutaneous Infection | 2,103 | Hyperkeratotic verrucous papules with punctate black dots (thrombosed capillaries). |

*Split Methodology:* An in-memory **Stratified 80/20 Split** (via `StratifiedShuffleSplit`, seed=42) preserves exact class proportions across training (21,722 images) and validation (5,431 images).

---

## 5. DIGITAL IMAGE PROCESSING METHODOLOGY

```
+---------------------------------------------------------------------------------+
|                       DIGITAL IMAGE PROCESSING PIPELINE                         |
+---------------------------------------------------------------------------------+
| 1. INGESTION & COLOR NORMALIZATION                                              |
|    - Raw Image Ingestion (Webcam / File Upload) via PIL/OpenCV                  |
|    - Explicit BGR -> RGB Color Space Normalization                              |
|    - Canvas Resizing (232x232 Bilinear Interpolation)                           |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
| 2. STOCHASTIC DATA AUGMENTATION (TRAINING PIPELINE)                             |
|    - Random Crop (224x224) -> Translation Invariance                            |
|    - Random Horizontal Flip (p=0.5) -> Reflection Invariance                    |
|    - Random Rotation (degrees=15) -> Orientation Invariance                     |
|    - Color Jitter (Brightness/Contrast/Saturation=0.10, Hue=0.03) -> Light Invar.|
|    - Channel Normalization: ImageNet Mean [0.485,0.456,0.406], Std [0.229,...]  |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
| 3. DEEP CONVOLUTIONAL FEATURE EXTRACTION (ConvNeXt-Base BACKBONE)               |
|    - Pretrained ImageNet-1K Weights (ConvNeXt_Base_Weights.DEFAULT)             |
|    - 7x7 Depthwise Separable Convolutions (Large Receptive Field)               |
|    - Inverted Bottleneck Design (4x Channel Expansion in Hidden Layers)         |
|    - Layer Normalization (LayerNorm) & GELU Non-Linearity Activation            |
|    - Adaptive Global Average Pooling -> 1024-Dimensional Deep Feature Vector    |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
| 4. CLASSIFICATION & REGULARIZATION HEAD                                         |
|    - Linear Fully Connected Projection: 1024 -> 10 Dermatological Classes        |
|    - Label Smoothing Regularization (alpha = 0.1) in CrossEntropyLoss           |
|    - Softmax Activation -> Calibrated Visual Probability Vector P_image         |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
| 5. MULTIMODAL LATE FUSION CONSENSUS                                             |
|    - P_fused = (0.60 * P_image) + (0.40 * P_text)                               |
|    - Automated Safety Net: If max(P) < 0.40 -> Flag 'low_confidence'            |
+---------------------------------------------------------------------------------+
```

### 5.1 Image Preprocessing & Color Space Management
- **Color Fidelity:** Early OpenCV builds exhibited a subtle channel inversion bug where BGR arrays were written to disk after premature RGB conversions, flipping red erythema into blue artifacts. The revised pipeline enforces standard `RGB` loading via PIL and torchvision transforms, ensuring ImageNet normalization statistics align with the camera sensors.
- **Normalization:** Every image tensor is standardized using ImageNet RGB statistics:
  $$\hat{I}_c = \frac{I_c - \mu_c}{\sigma_c}, \quad \mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

### 5.2 Stochastic Data Augmentation
To combat the memorization of photographic lighting, background artifacts, and patient skin pigmentation, training images pass through real-time augmentations:
- **Random Crop (232 $\to$ 224 pixels):** Induces translation invariance so lesions are never locked to image centers.
- **Random Horizontal Flip ($p = 0.5$):** Reflects bilateral lesions without altering clinical identity.
- **Random Rotation ($\pm 15^\circ$):** Simulates handheld mobile camera tilt.
- **Color Jitter ($\pm 10\%$ brightness, contrast, saturation; $\pm 3\%$ hue):** Forces the model to identify cellular texture and border architecture rather than ambient room lighting.

### 5.3 ConvNeXt-Base Architecture Rationale
While classic architectures like ResNet-50 or EfficientNet-B0 are common, **ConvNeXt-Base** (Liu et al., 2022) modernizes standard CNNs with architectural designs borrowed from Vision Transformers (ViTs):
1. **$7 \times 7$ Depthwise Convolutions:** Expands the effective receptive field per layer from $3 \times 3$ to $7 \times 7$, matching the visual attention span of Swin Transformers while preserving convolutional translation equivariance.
2. **Inverted Bottleneck Design:** Expands hidden channel dimensions by $4\times$ before depthwise projection, maximizing feature representation without exploding parameter counts.
3. **LayerNorm over BatchNorm:** Eliminates batch-size dependency during normalization, preventing variance drift during multi-GPU distributed data parallel passes.
4. **GELU Non-Linearity:** Gaussian Error Linear Units provide smooth gradient propagation across negative inputs compared to standard clipped ReLUs.

### 5.4 Distributed Multi-GPU Training (PyTorch DDP)
Training is executed on **Kaggle Notebooks** across 2x NVIDIA Tesla T4 GPUs (16GB VRAM each):
- **Communication Protocol:** PyTorch NCCL (`torch.distributed`) with `torch.multiprocessing.set_start_method("spawn")`.
- **Automatic Mixed Precision (AMP FP16):** Tensor operations execute in half-precision, doubling memory bandwidth while `GradScaler` prevents gradient underflow.
- **Per-GPU Batch Allocation:** Global batch size of 128 (64 per GPU), managed by `DistributedSampler` with deterministic per-epoch seeds.
- **Host RAM Protection:** Pinned host memory (`pin_memory=True`), defensive `malloc_trim()`, and distributed RAM surveillance tripwires prevent Kaggle's 30 GiB host memory ceiling from triggering kernel terminations.

---

## 6. SUPPORTING SYSTEM INFRASTRUCTURE

While DIP serves as the primary visual engine, the end-to-end clinical platform is supported by full-stack infrastructure:

```
+----------------------------------------------------------------------------+
|                       SUPPORTING MULTIMODAL SUBSYSTEMS                      |
+----------------------------------------------------------------------------+
| 1. CLINICAL NLP (Bio_ClinicalBERT):                                        |
|    - 110M parameter clinical transformer fine-tuned on 8,000 symptom notes. |
|    - 80.00% Accuracy on held-out test split, 0.86 recall on BCC.           |
|                                                                            |
| 2. LATE DECISION FUSION (fuse.py):                                         |
|    - Mathematical consensus: P_fused = (0.60 * P_image) + (0.40 * P_text)  |
|    - Dynamic single-modality fallback (text-only or image-only).           |
|                                                                            |
| 3. PERSISTENCE & SECURITY (SQLite & PBKDF2):                               |
|    - Salted PBKDF2-HMAC-SHA256 password security (260,000 iterations).      |
|    - Longitudinal health tracker analyzing linear trends across 5 sessions.|
|                                                                            |
| 4. MULTI-FORMAT I/O & NOTIFICATIONS:                                       |
|    - Parsers (.txt/.csv/.pdf/.docx) & Exporters (ReportLab PDF / Word DOCX)|
|    - Speech-to-Text (STT) & Google Text-to-Speech (gTTS) audio synthesis.  |
|    - APScheduler follow-ups via Twilio SMS and Gmail SMTP email relay.     |
+----------------------------------------------------------------------------+
```

---

## 7. EXPERIMENTAL RESULTS & PERFORMANCE HIGHLIGHTS

### ConvNeXt-Base Training Trajectory (Kaggle T4 x2)

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.1495 | 69.74% | 0.9472 | 78.90% | Initial Convergence |
| 2 | 0.9099 | 81.24% | 0.8610 | 83.27% | Rapid Feature Extraction |
| 3 | 0.8119 | 86.18% | 0.8196 | 85.90% | Exceeded 85% Target |
| 4 | 0.7384 | 89.69% | 0.8136 | 87.04% | Robust Progress |
| 5 | 0.6760 | 92.74% | 0.7905 | 88.09% | Loss Stabilization |
| 6 | 0.6359 | 94.62% | 0.7797 | 89.32% | Minimum Val Loss |
| 7 | 0.6073 | 96.08% | 0.7895 | 88.81% | High-Capacity Generalization |
| 8 | 0.5881 | 96.88% | 0.7831 | 89.43% | Smooth Annealing |
| **9** | **0.5778** | **97.41%** | **0.7822** | **89.65%** | **BEST CHECKPOINT (Saved)** |
| 10 | 0.5707 | 97.71% | 0.7806 | 89.56% | Final Convergence (51.55 min) |

### Key Experimental Findings:
1. **Surpassed Academic Requirement:** ConvNeXt-Base achieved **89.65% Validation Accuracy** at Epoch 9, surpassing the 85.0% baseline requirement stipulated by the course lecturer by **+4.65%**.
2. **Transfer Learning Superiority:** Upgrading from EfficientNet-B0 (which plateaued at 83.76%) to ConvNeXt-Base yielded an impressive **+5.89% absolute accuracy improvement**, demonstrating the benefits of $7 \times 7$ receptive fields for dermatological textures.
3. **Robust Generalization & Zero Loss Divergence:** Training accuracy (97.41%) and validation accuracy (89.65%) exhibited an optimal **7.76% generalization gap**, with validation loss remaining flat (~0.78) across epochs 6–10, confirming that Label Smoothing ($\alpha=0.1$) and Weight Decay ($\lambda=0.01$) prevented memorization.
4. **Permanent Host RAM Leak Resolution:** The implementation of pinned host memory and persistent distributed surveillance maintained host RAM usage flat at **~2.5 GiB** (leaving >25.8 GiB available), permanently eliminating the zero-swap Kaggle memory leak.

---

## 8. HARDWARE & SOFTWARE REQUIREMENTS

### Software Requirements
- **Operating System:** Linux (Pop!_OS 24.04 LTS x86_64)
- **Programming Language:** Python 3.12.3
- **Deep Learning Frameworks:** PyTorch 2.6.0 / 2.10.0+cu128, Torchvision 0.21.0
- **Computer Vision & Math Libraries:** OpenCV-Python (`cv2`), NumPy, PIL (Pillow), Scikit-Learn
- **Distributed Training Engine:** PyTorch NCCL via `torchrun` (2 processes)
- **Application & Presentation:** Streamlit, ReportLab (PDF), Python-Docx (DOCX), gTTS, SpeechRecognition
- **Database & Scheduling:** SQLite3, APScheduler, Twilio SDK

### Hardware Requirements
- **Local Development / Inference:**
  - Processor: 11th Gen Intel® Core™ i5-1135G7 @ 2.40GHz (8 threads)
  - Integrated GPU: Intel® Iris® Xe Graphics (Host CPU RAM inference)
  - Memory: 16 GB DDR4 RAM
  - External Video Input: Mobile phone camera connected via USB-C (DroidCam / Iriun webcam mode)
- **Cloud Training Environment:**
  - Platform: Kaggle Notebooks (GPU T4 x2)
  - Accelerators: 2x NVIDIA Tesla T4 GPUs (16 GB GDDR6 VRAM each, 32 GB total)
  - Host RAM: 30 GiB Linux Virtual Environment (Usage stabilized at ~2.5 GiB)

---

## 9. CONCLUSION & FUTURE SCOPE

The Digital Image Processing branch of this project successfully delivers a production-grade dermatological vision system. By leveraging ConvNeXt-Base, ImageNet transfer learning, stochastic data augmentations, and distributed GPU training, the vision model achieved **89.65% validation accuracy** across ten complex skin conditions. When unified with the companion Clinical NLP branch via Late Fusion, the system provides balanced, reliable, and accessible teledermatology support.

### Future Scope:
1. **Dermoscopic Segmentation Masks:** Implement a U-Net or Mask R-CNN segmentation pre-filter to isolate lesion boundaries and eliminate background healthy skin from visual feature extraction.
2. **Mobile Edge Quantization:** Quantize ConvNeXt-Base weights from FP32 to 4-bit INT4 via ONNX Runtime / TensorRT, enabling real-time offline inference on Android edge devices in rural clinics without cloud connectivity.
3. **Dermoscopy vs. Clinical Dual-Head Models:** Expand visual training to accept both standard smartphone photographs and polarized dermatoscope lens attachments through a dual-attention input head.

---

## 10. REFERENCES

1. **Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022).** A convnet for the 2020s. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 11976–11986.
2. **Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017).** Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542(7639), 115–118.
3. **Tschandl, P., Rosendahl, C., & Kittler, H. (2018).** The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5(1), 1–9.
4. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778.
5. **Loshchilov, I., & Hutter, F. (2017).** Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*.
6. **Müller, R., Kornblith, S., & Hinton, G. E. (2019).** When does label smoothing help?. *Advances in Neural Information Processing Systems (NeurIPS)*, 32, 4694–4703.

---

**Signature of Candidate:**  
Pratheek Poojari

**Signature of Guide:**  
Mrs. Madhushree B S  
*(Assistant Professor, Surana College)*
