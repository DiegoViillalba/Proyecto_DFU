# Reporte de Entrenamiento de Prueba - Proyecto DFU

Este documento detalla los resultados obtenidos en las ejecuciones de entrenamiento de prueba (clean runs) realizadas para verificar la correcta integración de las nuevas rutas de datos relativas, la exclusión en `.gitignore` y el versionamiento con DVC.

---

## 1. Configuración General del Entrenamiento
Para validar el flujo completo de datos sin incurrir en tiempos excesivos de GPU Blackwell RTX 5070, los entrenamientos se configuraron con los siguientes hiperparámetros de control:
* **Número de Épocas:** 1
* **Tamaño de Batch:** 4
* **Optimizador:** Adam
* **Tasa de Aprendizaje (Learning Rate):** 1e-5
* **Resolución de Imagen:** 240x240 píxeles

---

## 2. Resultados de los Modelos Entrenados

### A. Modelo de Clasificación: ResUnet
* **Ubicación del Código:** [src/ClasificationAlgorithms/Models/ResUnet/](file:///home/diego-villalba/Proyecto_DFU/src/ClasificationAlgorithms/Models/ResUnet/)
* **Conjunto de Datos:** `data/dfu_tissue/` (Dummy tissue classification set).
* **Métricas Obtenidas (Época 1):**
  * **Pérdida (Loss):** 0.8340
  * **Mean Dice Coefficient:** 0.0532
  * **Dice Coeff por Clase [0, 1, 2, 3]:** `['0.0000', '0.2131', '0.0000', '0.0000']`
  * **Precisión (Accuracy) por Clase:** `['0.2332', '0.1193', '0.8861', '1.0000']`
* **Archivos Generados:**
  * Checkpoint de pesos: `output_assets_model/best_model_checkpoint_ResUnet.pth` (y su versión comprimida `.zip`).
  * Historial de métricas: `output_assets_model/metrics_per_epoch_ResUnet.csv`.
  * Gráfico de curvas de pérdida: `output_assets_model/dice_loss_graph.png`.

### B. Modelo de Segmentación: ResUnet
* **Ubicación del Código:** [src/SegmentationNetworks/Models/ResUnet/](file:///home/diego-villalba/Proyecto_DFU/src/SegmentationNetworks/Models/ResUnet/)
* **Conjunto de Datos:** `data/miccai/` (MICCAI Dataset - 858 imágenes reales).
* **Métricas Obtenidas (Época 1):**
  * **Dice Score (Coeficiente Dice):** 0.5053
  * **Precisión General (Accuracy):** 98.551 %
* **Archivos Generados:**
  * Checkpoint de pesos: `output_assets_model/best_model_checkpoint_ResUnet.pth` (y su versión comprimida `.zip`).
  * Historial de métricas: `output_assets_model/metrics_ResUnet.csv`.
  * Gráfico de curvas de pérdida: `output_assets_model/dice_loss_graph_ResUnet.png`.
  * Parámetros JSON: `output_assets_model/parameters_ResUnet.json`.

---

## 3. Estructura y Orden de los Checkpoints
De acuerdo con el diseño de arquitectura propuesto, cada modelo almacena sus pesos y métricas localmente dentro de su propia subcarpeta en un directorio denominado `output_assets_model/`. Esto mantiene las métricas de clasificación y segmentación de cada red separadas y previene sobrescribirse accidentalmente.

Ejemplo de estructura de salida:
```text
src/SegmentationNetworks/Models/ResUnet/
└── output_assets_model/
    ├── best_metrics_val(during_training)_ResUnet.csv
    ├── best_model_checkpoint_ResUnet.pth   <-- Checkpoint de pesos pesados
    ├── best_model_checkpoint_ResUnet.zip   <-- Copia de seguridad comprimida
    ├── dice_loss_graph_ResUnet.png
    ├── metrics_ResUnet.csv
    ├── parameters_ResUnet.csv
    └── parameters_ResUnet.json
```

---

## 4. Política de Exclusión y Control de Espacio (.gitignore)
Para mantener el repositorio Git ligero y evitar subir checkpoints de gran tamaño (98 MB cada uno) a la nube, se ha implementado una política estricta en [.gitignore](file:///home/diego-villalba/Proyecto_DFU/.gitignore):
1. **Exclusión de carpetas de salida:** Se ignora recursivamente cualquier carpeta `output_assets_model/` en cualquier nivel mediante la regla `**/output_assets_model/`.
2. **Exclusión por extensión:** Se ignoran archivos con extensiones `*.pth`, `*.pth.tar`, `*.zip` y `*.mat` para evitar que subidas accidentales de binarios pesados saturen el almacenamiento de Git.
3. **Control por DVC:** Las imágenes pesadas de los datasets están delegadas en `data/` y controladas únicamente por DVC, mientras que Git solo trackea los punteros `.dvc`.
