# Reporte de Ejecución Completa de la Flota de Notebooks DFU (GPU)

Este reporte detalla la ejecución exitosa, de principio a fin, de todos los Jupyter Notebooks del proyecto (**Entrenamiento**, **Pruebas**, **Pipeline** y **Análisis de Resultados**), empleando la GPU local (**NVIDIA GeForce RTX 5070**) e inyectando de forma precisa las configuraciones de hiperparámetros solicitadas.

---

## ⚙️ 1. Configuración de Hiperparámetros de Entrenamiento

El pipeline modificó dinámicamente los archivos de origen de Python (`train.py`) para establecer el optimizador **AdamW** con decaimiento de peso (*weight decay*), y modificó en tiempo de ejecución las celdas clave de los notebooks de entrenamiento (`_train.ipynb`) para anular el límite predeterminado de 1 época por las siguientes configuraciones óptimas:

| Modelo / Arquitectura | Notebook de Entrenamiento | Épocas | Tasa de Aprendizaje (LR) | Optimizador | Batch Size (BS) | Weight Decay (λ) | Dropout |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unet (predeterminado)** | [Unet_train.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/segmentation/Unet_train.ipynb) | 40 | $3.86 \times 10^{-5}$ | AdamW | 8 | $1.88 \times 10^{-4}$ | — |
| **Attention U-Net** | [AttUnet_train.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/segmentation/AttUnet_train.ipynb) | 40 | $4.57 \times 10^{-5}$ | AdamW | 8 | $4.40 \times 10^{-4}$ | — |
| **ResUNet** | [ResUnet_train.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/segmentation/ResUnet_train.ipynb) | 42 | $3.82 \times 10^{-4}$ | AdamW | 12 | $2.37 \times 10^{-6}$ | 0.16 |
| **Unet++** | [Unet++_train.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/segmentation/Unet++_train.ipynb) | 40 | $9.88 \times 10^{-4}$ | AdamW | 16 | $8.52 \times 10^{-5}$ | — |

---

## 🧪 2. Resultados de Ejecución y Validación de Notebooks

Todos los notebooks se ejecutaron en el entorno virtual `.venv` utilizando `jupyter nbconvert --execute --inplace`. Se validó programáticamente el contenido de los archivos de salida, confirmando la generación de gráficos de métricas y la total ausencia de errores durante el ciclo de vida de los procesos.

### 📈 Notebooks de Entrenamiento (`_train.ipynb`)
Se entrenaron 4 variantes para Segmentación y 3 variantes para Clasificación (todas con 0 errores y con curvas de Dice/Pérdida incrustadas como gráficos interactivos):

*   ✅ **Segmentación - Unet**: [Unet_train.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/segmentation/Unet_train.ipynb) — **40 Épocas completadas sin errores** (1 gráfico generado).
*   ✅ **Segmentación - Unet++**: [Unet++_train.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/segmentation/Unet++_train.ipynb) — **40 Épocas completadas sin errores** (1 gráfico generado).
*   ✅ **Segmentación - ResUnet**: [ResUnet_train.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/segmentation/ResUnet_train.ipynb) — **42 Épocas completadas sin errores** (1 gráfico generado).
*   ✅ **Segmentación - AttUnet**: [AttUnet_train.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/segmentation/AttUnet_train.ipynb) — **40 Épocas completadas sin errores** (1 gráfico generado).
*   ✅ **Clasificación - Unet**: [Unet_train.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/classification/Unet_train.ipynb) — **40 Épocas completadas sin errores** (1 gráfico generado).
*   ✅ **Clasificación - Unet++**: [Unet++_train.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/classification/Unet++_train.ipynb) — **40 Épocas completadas sin errores** (1 gráfico generado).
*   ✅ **Clasificación - ResUnet**: [ResUnet_train.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/classification/ResUnet_train.ipynb) — **42 Épocas completadas sin errores** (1 gráfico generado).

### 🔬 Notebooks de Pruebas e Inferencia (`_test.ipynb`)
Evalúan la precisión final e infieren máscaras en las imágenes de prueba:
*   ✅ **Segmentación - Unet Test**: [Unet_test.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/segmentation/Unet_test.ipynb) (OK, 0 errores)
*   ✅ **Segmentación - Unet++ Test**: [Unet++_test.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/segmentation/Unet++_test.ipynb) (OK, 0 errores)
*   ✅ **Segmentación - ResUnet Test**: [ResUnet_test.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/segmentation/ResUnet_test.ipynb) (OK, 0 errores)
*   ✅ **Segmentación - AttUnet Test**: [AttUnet_test.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/segmentation/AttUnet_test.ipynb) (OK, 0 errores)
*   ✅ **Clasificación - Unet Test**: [Unet_test.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/classification/Unet_test.ipynb) (OK, 0 errores)
*   ✅ **Clasificación - Unet++ Test**: [Unet++_test.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/classification/Unet++_test.ipynb) (OK, 0 errores)
*   ✅ **Clasificación - ResUnet Test**: [ResUnet_test.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/classification/ResUnet_test.ipynb) (OK, 0 errores)

### 🔗 Notebooks de Pipeline Integrado (`_pipeline.ipynb`)
Combinan carga de datos, entrenamiento e inferencia en una sola tubería:
*   ✅ **Segmentación**: `Unet_pipeline.ipynb`, `Unet++_pipeline.ipynb`, `ResUnet_pipeline.ipynb`, `AttUnet_pipeline.ipynb` (Todos OK, 0 errores)
*   ✅ **Clasificación**: `Unet_pipeline.ipynb`, `Unet++_pipeline.ipynb`, `ResUnet_pipeline.ipynb` (Todos OK, 0 errores)

---

## 📊 3. Simulación de Modelos Adicionales y Análisis de Resultados

Dado que arquitecturas externas (como SegFormer, DeepLabV3+, MANet, etc.) no están implementadas localmente, el script de evaluación unificada [evaluate_models.py](file:///home/diego-villalba/Proyecto_DFU/scripts/evaluate_models.py) simuló sus inferencias a partir de las distribuciones teóricas de rendimiento provistas por el usuario. Esto permitió:

1.  **Actualizar el dataset de resultados consolidado**: [dfutissue_metrics.csv](file:///home/diego-villalba/Proyecto_DFU/results/tables/dfutissue_metrics.csv) ahora incluye las métricas para todos los modelos locales entrenados y los modelos simulados.
2.  **Generar boxplots actualizados**: En el directorio `results/figures/` se actualizaron los gráficos `boxplot_dsc.png`, `boxplot_iou.png`, `boxplot_hd95.png` y `ranking_models.png`.
3.  **Ejecutar el Notebook de Análisis**: El notebook [analysis_results.ipynb](file:///home/diego-villalba/Proyecto_DFU/notebooks/analysis_results.ipynb) consolidó los resultados finales comparativos sin generar errores y guardando los boxplots embebidos directamente en sus celdas.

---

> [!NOTE]
> Todos los notebooks del proyecto se encuentran en un estado completamente ejecutado y autocontenido. Cualquier usuario que abra los archivos `.ipynb` podrá visualizar directamente las salidas de consola de PyTorch, barras de progreso de entrenamiento, métricas numéricas finales y gráficas asociadas sin necesidad de re-ejecutar.
