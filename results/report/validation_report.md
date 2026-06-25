# Quantitative and Qualitative Validation of Deep Learning Segmentation Models on the DFUTissue Dataset

**Author:** Diego Villalba et al.  
**Affiliation:** DFUTissue Segmentation Research Group  
**Date:** June 25, 2026  

---

## Abstract
Accurate segmentation of diabetic foot ulcer (DFU) tissue types—specifically granulation, slough, and necrotic tissues—is a vital task for clinical monitoring, treatment selection, and wound healing tracking. This report presents a quantitative evaluation and comparison of five state-of-the-art segmentation architectures (**MANet MiT-b3**, **U-Net MiT-b3**, **SegFormer MiT-b3**, **ResUNet**, and **U-Net MobileNetV2**) evaluated on the **DFUTissue** test set. Over a 20-image cohort, per-image metric analysis (Dice Similarity Coefficient, Intersection over Union, Precision, Recall, 95th Percentile Hausdorff Distance, and Average Symmetric Surface Distance) was performed. Furthermore, a qualitative evaluation was conducted by automatically selecting and visualizing the best, average, and difficult cases using the best-performing model, complete with pixel-level error maps. The results show a clear performance hierarchy, with transformer-based encoders achieving exceptional contour matching and surface accuracy.

---

## 1. Introduction
Diabetic foot ulcers (DFUs) represent a severe complication of diabetes mellitus, often leading to lower-extremity amputations if unmanaged. The healing progress is closely correlated with the relative distribution of wound tissues:
- **Granulation Tissue (Class 1):** Indicates active healing and vascularization (pink/red color).
- **Slough (Class 2):** Composed of dead cellular debris, requiring debridement (yellow/white color).
- **Necrotic Tissue (Class 3):** Dead tissue indicating severe ischemia, requiring immediate clinical intervention (black/dark color).

Automated computer-aided segmentation utilizing deep convolutional neural networks (CNNs) and Vision Transformers (ViTs) offers a standardized, objective, and reproducible evaluation of wound progress. This report documents both the quantitative and qualitative validation phases of five models to establish a benchmark for clinical applications.

---

## 2. Materials and Methods

### 2.1 Dataset and Image Preprocessing
The **DFUTissue** test set consists of 20 high-resolution clinical images and their corresponding pixel-level annotations (ground-truth masks) classifying pixels into four labels:
- **Class 0:** Background / Healthy Perilesional Skin
- **Class 1:** Granulation Tissue
- **Class 2:** Slough
- **Class 3:** Necrotic Tissue

All images were padded and resized to $240 \times 240$ pixels to ensure consistent input dimensions across all models.

### 2.2 Segmented Models Under Analysis
The evaluated models represent a mixture of heavy encoder-decoder transformer architectures and lighter, speed-optimized CNN architectures:
1. **MANet MiT-b3:** Multi-scale Attention Net with a Mix Vision Transformer (MiT-b3) backbone.
2. **U-Net MiT-b3:** Standard U-Net structure utilizing a Mix Vision Transformer (MiT-b3) encoder.
3. **SegFormer MiT-b3:** Lightweight transformer-based segmentation architecture.
4. **ResUNet:** Residual U-Net featuring a ResNet-34 encoder.
5. **U-Net MobileNetV2:** Lightweight U-Net with a MobileNetV2 encoder designed for mobile execution.

### 2.3 Evaluation Metrics
To measure the overlap, boundary quality, and spatial error of the models, six quantitative metrics were calculated per image and macro-averaged over the foreground classes ($1$, $2$, and $3$):

1. **Dice Similarity Coefficient (DSC):** Measures overlap area.
   $$\text{DSC} = \frac{2 |Y \cap \hat{Y}|}{|Y| + |\hat{Y}|}$$
2. **Intersection over Union (IoU):** Jaccard index measuring area overlap.
   $$\text{IoU} = \frac{|Y \cap \hat{Y}|}{|Y \cup \hat{Y}|}$$
3. **Precision:** Ratio of correctly predicted positive pixels to total predicted positive pixels.
4. **Recall (Sensitivity):** Ratio of correctly predicted positive pixels to actual positive pixels.
5. **95% Hausdorff Distance (HD95):** The 95th percentile of the maximum distance between predicted and ground-truth boundary surfaces, avoiding outlier sensitivity.
6. **Average Symmetric Surface Distance (ASSD):** The average distance between the boundary surface of the prediction and the ground truth.

---

## 3. Results (Quantitative)

Quantitative results for the 20-image cohort are summarized in the table below (values represent the mean of the 20 test images):

| Model | DSC ↑ | IoU ↑ | Precision ↑ | Recall ↑ | HD95 (px) ↓ | ASSD (px) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MANet MiT-b3** | 0.9856 | 0.9718 | 0.9795 | 0.9922 | 1.00 px | 0.54 px |
| **U-Net MiT-b3** | 0.9818 | 0.9644 | 0.9734 | 0.9908 | 1.00 px | 0.68 px |
| **ResUNet** | 0.9673 | 0.9375 | 0.9508 | 0.9861 | 1.84 px | 0.97 px |
| **SegFormer MiT-b3** | 0.9649 | 0.9330 | 0.9492 | 0.9833 | 1.91 px | 1.05 px |
| **U-Net MobileNetV2** | 0.9492 | 0.9049 | 0.9258 | 0.9778 | 2.81 px | 1.30 px |

---

## 4. Discussion (Quantitative)

### 4.1 Quantitative Performance and Hierarchy
The quantitative analysis demonstrates a distinct division in performance based on model complexity and architecture:
*   **Transformer-based Encoders (MiT-b3):** Both **MANet MiT-b3** (DSC: **0.9856**) and **U-Net MiT-b3** (DSC: **0.9818**) show excellent performance. Their attention mechanisms allow them to capture multi-scale global contexts, which are crucial for delineating complex boundaries in ulcer tissues.
*   **Residual CNNs vs. Transformers:** The **ResUNet** (DSC: **0.9673**, HD95: **1.84 px**) represents a middle-ground model, striking a balance between local spatial features from CNNs and feature reuse through residual connections. It outperforms the lightweight **SegFormer MiT-b3** (DSC: **0.9649**).
*   **Mobile-Optimized Models:** **U-Net MobileNetV2** exhibits the lowest overall scores (DSC: **0.9492**, IoU: **0.9049**). However, its performance remains clinically acceptable while requiring a fraction of the computational resources, making it suitable for edge-device integration.

### 4.2 Surface Distance Significance (HD95 & ASSD)
While area-based metrics (DSC/IoU) are highly useful, boundary surface metrics (HD95 and ASSD) provide critical insight into clinical accuracy. **MANet MiT-b3** achieved a low ASSD error of (**0.54 px**), meaning that its boundary deviations from the ground-truth annotations are less than a pixel on average. In contrast, **U-Net MobileNetV2** had an ASSD error of **1.30 px**, which indicates higher susceptibility to irregular border predictions.

---

## 5. Qualitative Results (Visual Evidence)

To validate the model’s clinical effectiveness and map spatial errors, a qualitative selection was automated using the best-performing model (**MANet MiT-b3**), which was dynamically selected by the notebook based on the maximum mean DSC. The test cohort was sorted by DSC, and three difficulty categories were established:
1. **Best Cases (DSC: 0.990 - 0.995):** Perfect tissue delineation with minimal boundary noise.
2. **Average Cases (DSC: 0.982 - 0.985):** Excellent overlap, with minor discrepancies on overlapping tissue boundaries.
3. **Difficult Cases (DSC: 0.945 - 0.960):** Complex tissue mixtures showing slight under-segmentation in boundary margins.

### 5.1 Case-by-Case Visual Comparisons
For each case, a comparative panel was created containing:
- **Original RGB Image:** Visual color features.
- **Ground Truth:** Clinical target labels.
- **Best Model Prediction:** Output mask.
- **Error Map:** Highlighting correct background (dark gray), correct foreground (light gray), and mismatch pixels (bright red).

All 15 individual panels are saved and accessible in the folder [results/figures/qualitative/](file:///home/diego-villalba/Proyecto_DFU/results/figures/qualitative/).

### 5.2 Qualitative Summary Panel
A comparative summary grid was built to display one representative case from each category (Best, Average, and Difficult). This panel highlights how the MANet MiT-b3 model performs under different wound complexities. The summary panel is stored at [results/figures/qualitative_summary.png](file:///home/diego-villalba/Proyecto_DFU/results/figures/qualitative_summary.png).

---

## 6. Conclusion
In summary, transformer-augmented segmentation networks (**MANet MiT-b3** and **U-Net MiT-b3**) provide the most accurate quantitative boundary and overlap performance for DFU tissue categorization. The visual evidence confirms that boundary errors are sub-pixel on average in easy cases, and limited to narrow transition margins in difficult cases. Lightweight CNNs (**U-Net MobileNetV2**) present a viable alternative for fast mobile-device inference, though they compromise slightly on boundary precision. Future work will incorporate statistical hypothesis testing (e.g., Wilcoxon signed-rank tests) to establish the statistical significance of these performance differences.

---

### Reference Figures and Tables
- **Dice Similarity Distribution:** See [boxplot_dsc.png](file:///home/diego-villalba/Proyecto_DFU/results/figures/boxplot_dsc.png)
- **Intersection over Union Distribution:** See [boxplot_iou.png](file:///home/diego-villalba/Proyecto_DFU/results/figures/boxplot_iou.png)
- **Hausdorff Distance 95 Distribution:** See [boxplot_hd95.png](file:///home/diego-villalba/Proyecto_DFU/results/figures/boxplot_hd95.png)
- **Performance Ranking Visual:** See [ranking_models.png](file:///home/diego-villalba/Proyecto_DFU/results/figures/ranking_models.png)
- **Qualitative Summary Grid:** See [qualitative_summary.png](file:///home/diego-villalba/Proyecto_DFU/results/figures/qualitative_summary.png)
- **Individual Qualitative Comparisons:** Accessible in [results/figures/qualitative/](file:///home/diego-villalba/Proyecto_DFU/results/figures/qualitative/)
- **Metrics Dataset:** Detailed per-image metrics are exported to [results/tables/dfutissue_metrics.csv](file:///home/diego-villalba/Proyecto_DFU/results/tables/dfutissue_metrics.csv).
