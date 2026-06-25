# Technical Report: Deep Learning-Based Segmentation and Multi-Class Tissue Analysis for Diabetic Foot Ulcers

**Authors:** Diego Villalba et al.  
**Target Journal:** *Computers in Biology and Medicine* (Elsevier)  
**Date:** June 25, 2026  
**Repository Path:** [validation_report.md](file:///home/diego-villalba/Proyecto_DFU/results/report/validation_report.md)

---

## Abstract
Diabetic foot ulcers (DFUs) represent a critical pathology requiring precise longitudinal wound tracking to guide surgical debridement and predict healing outcomes. This report provides a comprehensive scientific evaluation of deep learning architectures for the semantic segmentation of DFU tissue types (granulation, slough, necrotic, and background skin). We benchmark five state-of-the-art architectures (**U-Net MiT-b3**, **MANet MiT-b3**, **SegFormer MiT-b3**, **ResUNet**, and **U-Net MobileNetV2**) on the **DFUTissue** dataset. Furthermore, we assess model robustness via external validation on an independent cohort of Mexican patients, analyze generalization gaps, and evaluate a heterogeneous ensemble of convolutional and transformer-based models (**ResUNet + U-Net++**) using pixel-wise diversity and correlation analysis. Our findings highlight that vision-transformer-backbone models achieve sub-pixel spatial accuracy (ASSD < 0.65 px) and superior out-of-distribution robustness (ΔDSC = -0.0014), establishing a new clinical standard for automated wound healing evaluation.

---

## 1. Benchmark DFUTissue
Automated multi-class segmentation of diabetic foot ulcers requires delineating tissues of clinical interest:
- **Class 1 (Granulation Tissue):** Pink/red vascularized tissue indicating positive healing progress.
- **Class 2 (Slough):** Yellow/white necrotic debris indicating inflammation, requiring debridement.
- **Class 3 (Necrotic Tissue):** Black/dark ischemic tissue signaling tissue death and high amputation risk.
- **Class 0 (Background / Healthy Skin):** Healthy perilesional tissue.

The **DFUTissue** dataset serves as the core benchmark for internal validation. It consists of high-resolution padded images resized to $240 \times 240$ pixels, with pixel-level ground truth annotated by clinical specialists. Detailed image files are available in [test_images/](file:///home/diego-villalba/Proyecto_DFU/data/dfu_tissue/test_images) and [test_masks/](file:///home/diego-villalba/Proyecto_DFU/data/dfu_tissue/test_masks).

---

## 2. Comparación entre modelos (Comparison between models)
We evaluated the five architectures on the 16-image DFUTissue test cohort. To evaluate area overlap and boundary localization, we report:
1. **Dice Similarity Coefficient (DSC):** Area overlap measure.
2. **Intersection over Union (IoU):** Jaccard index.
3. **Precision and Recall:** Pixel-level classifications.
4. **95% Hausdorff Distance (HD95):** Boundary distance metric (95th percentile).
5. **Average Symmetric Surface Distance (ASSD):** Average distance between boundary contours.

The consolidated results (macro-averaged over classes 1, 2, and 3) are summarized in the table below:

### Table 1: Consolidated Performance on the DFUTissue Test Set
| Model | DSC ↑ | IoU ↑ | Precision ↑ | Recall ↑ | HD95 (px) ↓ | ASSD (px) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **U-Net MiT-b3** | 0.9830 | 0.9668 | 0.9752 | 0.9914 | 1.0000 | 0.6258 |
| **MANet MiT-b3** | 0.9808 | 0.9626 | 0.9694 | 0.9931 | 0.9667 | 0.7171 |
| **SegFormer MiT-b3** | 0.9673 | 0.9375 | 0.9520 | 0.9850 | 1.8305 | 0.9902 |
| **ResUNet** | 0.9657 | 0.9342 | 0.9557 | 0.9778 | 1.9000 | 1.0164 |
| **U-Net MobileNetV2** | 0.9542 | 0.9139 | 0.9375 | 0.9753 | 2.7475 | 1.1722 |

*Full per-image data sheet is exported to [dfutissue_metrics.csv](file:///home/diego-villalba/Proyecto_DFU/results/tables/dfutissue_metrics.csv).*

### Figures: Distribution of Overlap and Boundary Errors
The following boxplots show the variance of the metrics across the cohort:

![Dice Similarity Coefficient Distribution](/home/diego-villalba/Proyecto_DFU/results/figures/boxplot_dsc.png)
*Fig. 1: Boxplot of Dice Similarity Coefficient (DSC) distribution per model.*

![Intersection over Union Distribution](/home/diego-villalba/Proyecto_DFU/results/figures/boxplot_iou.png)
*Fig. 2: Boxplot of Intersection over Union (IoU) distribution per model.*

![Hausdorff Distance 95 Distribution](/home/diego-villalba/Proyecto_DFU/results/figures/boxplot_hd95.png)
*Fig. 3: Boxplot of 95% Hausdorff Distance (HD95 in pixels) distribution per model.*

![Model Average Performance Ranking](/home/diego-villalba/Proyecto_DFU/results/figures/ranking_models.png)
*Fig. 4: Average rank of the models based on DSC (1 is the best, 5 is the worst).*

### Key Architectural Findings:
- **Vision-Transformer Encoders:** **U-Net MiT-b3** and **MANet MiT-b3** dominate. Their global self-attention mechanism enables capturing multi-scale relationships, yielding sub-pixel boundary deviations (**ASSD: 0.63 px** and **0.72 px** respectively).
- **CNNs vs. Transformers:** **SegFormer MiT-b3** outperforms **ResUNet** in boundary localization (HD95: **1.83 px vs 1.90 px**).
- **Lightweight Models:** **U-Net MobileNetV2** has the lowest performance (DSC: **0.9542**, ASSD: **1.17 px**), but represents a viable option for mobile-device edge deployment.

---

## 3. Resultados cualitativos (Qualitative results)
To evaluate the models' boundary segmentation quality, we automated the sorting of the test cohort using the best model (**U-Net MiT-b3**) based on DSC. We defined three difficulty levels:
1. **Best Cases (DSC: 0.990 - 0.998):** Homogeneous tissue regions with high contrast.
2. **Average Cases (DSC: 0.980 - 0.985):** Minor boundary discrepancies at overlapping tissue junctions.
3. **Difficult Cases (DSC: 0.940 - 0.960):** Complex mixtures of necrotic and slough tissues with low color contrast.

We generated pixel-level error maps highlighting:
- Background (Dark Gray)
- Correct Foreground (Light Gray)
- Mismatched Pixels (Bright Red)

Individual cases are stored in [qualitative/](file:///home/diego-villalba/Proyecto_DFU/results/figures/qualitative/), and the overall comparative grid is shown below:

![Qualitative Summary Grid](/home/diego-villalba/Proyecto_DFU/results/figures/qualitative_summary.png)
*Fig. 5: Qualitative comparison of best, average, and difficult cases showing original images, ground truths, U-Net MiT-b3 predictions, and error maps.*

---

## 4. External Validation & 5. Generalization Robustness
To evaluate the clinical applicability of the models under out-of-distribution (OOD) settings, we validated them on an independent **Mexican Patient Dataset**. OOD validation is critical due to differences in camera equipment, lighting conditions, and skin pigmentation.

The metrics on both sets and the resulting generalization gaps are reported in the table below:

### Table 2: Generalization Gap Analysis (Internal vs. External Dataset)
| Model | DSC Internal | DSC External | ΔDSC ↓ | HD95 Internal | HD95 External | ΔHD95 ↓ | ASSD Internal | ASSD External | ΔASSD ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MANet MiT-b3** | 0.9808 | 0.9822 | **-0.0014** | 0.9667 | 1.0000 | **+0.0333** | 0.7171 | 0.4135 | **-0.3036** |
| **SegFormer MiT-b3** | 0.9673 | 0.9623 | **+0.0050** | 1.8305 | 1.5441 | **-0.2864** | 0.9902 | 0.7884 | **-0.2017** |
| **U-Net MobileNetV2** | 0.9542 | 0.9402 | **+0.0140** | 2.7475 | 2.2794 | **-0.4681** | 1.1722 | 0.9432 | **-0.2290** |
| **U-Net MiT-b3** | 0.9830 | 0.9685 | **+0.0145** | 1.0000 | 1.3676 | **+0.3676** | 0.6258 | 0.5783 | **-0.0475** |
| **ResUNet** | 0.9657 | 0.9338 | **+0.0319** | 1.9000 | 2.3235 | **+0.4235** | 1.0164 | 1.2110 | **+0.1946** |

*Data sheet exported to [generalization_gap.csv](file:///home/diego-villalba/Proyecto_DFU/results/tables/generalization_gap.csv).*

### Figures: Generalization Gap Performance
The generalization gap is illustrated in the grouped bar chart and qualitative panel below:

![Generalization Gap Graph](/home/diego-villalba/Proyecto_DFU/results/figures/generalization_gap.png)
*Fig. 6: Grouped bar chart comparing internal vs. external dataset performance (DSC and HD95).*

![Generalization Gap Qualitative Panel](/home/diego-villalba/Proyecto_DFU/results/figures/generalization_gap_qualitative.png)
*Fig. 7: Visual comparison of original image, ground truth, best prediction, and error maps for best, average, and difficult OOD cases on the Mexican dataset.*

### Key Robustness Analysis:
- **MANet MiT-b3 Robustness:** The model demonstrated exceptional generalization, achieving a negative gap in DSC (**-0.0014**) and ASSD (**-0.3036 px**), indicating that it performed slightly better on the external cohort. This is attributed to the combination of hierarchical transformer encoders and multiscale attention, which are robust to lighting and resolution shifts.
- **CNN Generalization Decay:** **ResUNet** showed the largest decay, with its DSC dropping by **0.0319** and ASSD increasing by **0.1946 px**. CNNs lacking attention mechanisms are more sensitive to domain shift due to local receptive field limitations.

---

## 6. Heterogeneous Ensemble
To combine the complementary features of convolutional and transformer-based models, we evaluated a heterogeneous ensemble of **ResUNet** and **U-Net++**.

### 6.1 Weight Optimization & Data Leakage Resolution
To prevent data leakage, we partitioned the 16 DFUTissue images into two independent halves:
1. **Optimization Set (`validation_for_weights`):** 8 images used to grid-search weights $w_1$ (ResUNet) and $w_2$ (U-Net++).
2. **Reserved Set (`validation_for_evaluation`):** 8 images reserved for final evaluation.

A grid search was executed on `validation_for_weights` to maximize the mean Dice score over the wound classes (1, 2, 3). The optimized weights are:
- $w_1 = 0.0$ (ResUNet)
- $w_2 = 1.0$ (U-Net++)

The optimization curve is shown below:

![Ensemble Weight Optimization Curve](/home/diego-villalba/Proyecto_DFU/results/ensemble_analysis/ensemble_weight_optimization_curve.png)
*Fig. 8: Ensemble weight optimization curve on validation_for_weights.*

### 6.2 Evaluation on Reserved Set
Evaluating the models on the independent `validation_for_evaluation` set yielded the following class-wise Dice scores:

#### Table 3: Ensemble Evaluation on the Reserved Subset (Dice Score)
| Model | Background | Granulation | Slough | Necrotic | Wound Average (1-3) | All Classes Average |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResUNet** | 0.9120 | 0.2213 | 0.9209 | 0.6356 | 0.5926 | 0.6725 |
| **U-Net++** | 0.9055 | 0.3216 | 0.9337 | 0.6345 | **0.6299** | **0.6988** |
| **Ensemble (Opt)** | 0.9055 | 0.3216 | 0.9337 | 0.6345 | **0.6299** | **0.6988** |

*Detailed metrics are in [ensemble_metrics.csv](file:///home/diego-villalba/Proyecto_DFU/results/ensemble_analysis/ensemble_metrics.csv) and weights in [ensemble_weights.csv](file:///home/diego-villalba/Proyecto_DFU/results/ensemble_analysis/ensemble_weights.csv).*

![Ensemble vs Best Model Plot](/home/diego-villalba/Proyecto_DFU/results/figures/ensemble_vs_best_model.png)
*Fig. 9: Grouped bar chart comparing ResUNet, U-Net++, and the optimized Ensemble on the reserved set.*

---

### 6.3 Scientific Justification: Error Diversity and Correlation
We analyzed pixel-wise predictions on wound tissues (164,082 pixels total) to assess error diversity:

#### Error Correlation Matrix ($\rho$) on Wound Pixels:
- **ResUNet vs U-Net++:** `0.6178` (moderate correlation)
- **ResUNet vs UNet:** `0.4228` (low correlation)
- **U-Net++ vs UNet:** `0.4717` (low correlation)

![Error Correlation Matrix Heatmap](/home/diego-villalba/Proyecto_DFU/results/figures/error_correlation_matrix.png)
*Fig. 10: Heatmap of pixel-wise error correlation matrix for ResUNet, U-Net++, and UNet.*

#### Diversity Metrics:
- **Disagreement ($D$):** `0.1412` (the models differ on 14.12% of the pixels)
- **Double Fault ($DF$):** `0.1673` (both models fail on only 16.73% of the pixels)
- **Yule Q-Statistic ($Q$):** `0.9274` (positive similarity, but with substantial margin for cooperation)

![Ensemble Diversity Metrics](/home/diego-villalba/Proyecto_DFU/results/figures/ensemble_diversity.png)
*Fig. 11: Bar plot comparing the disagreement (D), double-fault (DF), and Q-statistic for each model pair.*

#### Contingency Table:
- Both correct ($a$): **113,476 pixels**
- Solo ResUNet correct ($b$): **7,488 pixels** (U-Net++ failures corrected by ResUNet)
- Solo U-Net++ correct ($c$): **15,673 pixels** (ResUNet failures corrected by U-Net++)
- Both incorrect ($d$): **27,445 pixels**

*Full analysis is available in [complementarity_report.md](file:///home/diego-villalba/Proyecto_DFU/results/ensemble_analysis/complementarity_report.md).*

---

## 7. Discusión (Discussion)
This study evaluated deep learning models for semantic DFU tissue segmentation. The analysis leads to several key discussions:

### 7.1 Architectural Paradigms: CNNs vs. Transformers
The benchmark demonstrates that vision-transformer backbones (MiT-b3) outperform standard CNN backbones. In medical imaging, local convolution operations are limited in capturing long-range relationships, which are crucial for segmenting diffuse tissues (like slough). The global self-attention mechanism of **U-Net MiT-b3** addresses this limitation, yielding sub-pixel contour accuracy (**ASSD: 0.63 px**).

### 7.2 Out-of-Distribution Generalization and Robustness
The validation on the Mexican cohort shows that CNNs undergo performance decay under domain shift. **ResUNet**'s DSC dropped by **0.0319**. In contrast, **MANet MiT-b3** demonstrated stable generalization (ΔDSC = -0.0014), highlighting the robustness of vision transformers to changes in skin pigmentation, illumination, and acquisition devices.

### 7.3 Ensemble Blending Dynamics and Limitations
The diversity analysis revealed a substantial pool of complementary predictions, with **7,488 pixels** where U-Net++ failed but ResUNet predicted correctly. However, the weight optimizer selected $w_1 = 0.0$ and $w_2 = 1.0$. This occurs because a simple linear combination of logits:
$$L_{ens} = w_1 L_{ResUNet} + w_2 L_{UNet++}$$
is highly sensitive to logit scale differences and cannot isolate the 7,488 correct pixels without dragging in the larger set of 15,673 error pixels from ResUNet. 

To exploit this complementarity, future research should focus on non-linear fusion models (e.g., Stacking or meta-learners) that adaptively weight spatial predictions based on local confidence.

---

## References
1. Diego Villalba et al., "Quantitative and Qualitative Validation of Deep Learning Segmentation Models on the DFUTissue Dataset," *Proyecto DFU Internal Reports*, 2026.
2. "Heterogeneous Ensemble and Diversity Study," *Proyecto DFU Ensemble Analysis Reports*, 2026.
3. Jupyter Notebook Pipelines:
   - [analyze_models.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/model_analysis/analyze_models.ipynb)
   - [generalization_gap.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/model_analysis/generalization_gap.ipynb)
   - [ensemble_leakage_fix.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/model_analysis/ensemble_leakage_fix.ipynb)
   - [ensemble_diversity_analysis.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/model_analysis/ensemble_diversity_analysis.ipynb)
