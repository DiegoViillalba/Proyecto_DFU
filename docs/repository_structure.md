# Estructura del Repositorio - Proyecto DFU

Este documento explica la organización y el propósito de cada directorio del proyecto de segmentación y clasificación de Úlceras de Pie Diabético (DFU). La estructura está diseñada para seguir las mejores prácticas de modularidad, reproducibilidad y orden en proyectos de Machine Learning.

---

## Árbol de Directorios

```text
repo/
├── configs/             # Archivos de configuración de hiperparámetros y entorno.
├── data/                # Datasets organizados y versionados con DVC.
│   ├── dfu_tissue/      # Dataset de clasificación/tejidos de DFU (train/val/test).
│   └── miccai/          # Dataset de segmentación MICCAI (train/val/test).
├── docs/                # Documentación técnica y diagramas de arquitectura.
│   └── architectures/   # Diagramas visuales de las redes neuronales.
├── notebooks/           # Cuadernos interactivos (Jupyter Notebooks) de experimentación.
│   └── segmentation/    # Notebooks de experimentación de modelos de segmentación.
├── results/             # Archivos de resultados y métricas del proyecto (.csv, .xlsx).
│   ├── classification/  # Métricas de los modelos de clasificación de tejidos.
│   └── segmentation/    # Métricas y hojas de cálculo de redes de segmentación.
├── scripts/             # Scripts ejecutables de automatización, utilidades y configuración.
├── src/                 # Código fuente del proyecto (Código Científico principal).
│   ├── ClasificationAlgorithms/ # Modelos y scripts para clasificación de tejidos (TissueSegNet).
│   └── SegmentationNetworks/    # Modelos y procesamiento para segmentación de DFU.
├── tests/               # Pruebas de código (tests unitarios y de integración).
└── weights/             # Pesos de modelos entrenados (.tflite, .pth, checkpoints).
```

---

## Descripción detallada de cada carpeta

### 📁 `configs/`
Destinada a albergar archivos de configuración estática o dinámica (por ejemplo, archivos `.json`, `.yaml` o `.ini`). Permite separar la lógica de programación del código científico de los hiperparámetros o rutas específicas de ejecución.

### 📁 `data/`
Este directorio centralizado almacena los conjuntos de datos (datasets) utilizados para el entrenamiento y prueba de los modelos. Para optimizar el espacio del repositorio Git y evitar guardar archivos binarios pesados (imágenes y máscaras), este directorio está completamente trackeado y versionado mediante **DVC (Data Version Control)**.

#### Estructura interna de `data/`:
*   **`data/miccai/`**: Dataset utilizado por las redes de segmentación en [src/SegmentationNetworks/](file:///home/diego-villalba/Proyecto_DFU/src/SegmentationNetworks/).
    *   `train_images/` y `train_masks/` (imágenes y máscaras binarias de entrenamiento).
    *   `val_images/` y `val_masks/` (imágenes y máscaras binarias de validación).
*   **`data/dfu_tissue/`**: Dataset de clasificación y segmentación multicapa de tejidos en [src/ClasificationAlgorithms/](file:///home/diego-villalba/Proyecto_DFU/src/ClasificationAlgorithms/).
    *   `train_images/` y `train_masks/` (imágenes y máscaras multiclase de entrenamiento).
    *   `val_images/` y `val_masks/` (imágenes y máscaras multiclase de validación).
    *   `test_images/` y `test_masks/` (imágenes y máscaras de prueba).

#### 🚀 Cómo agregar nuevas imágenes al Dataset y trackearlas con DVC
Para ampliar el conjunto de imágenes de entrenamiento de manera consistente y de fácil escala, sigue estos pasos:

1.  **Colocar los nuevos archivos**:
    Guarda las nuevas imágenes en la carpeta de imágenes adecuada (por ejemplo, `data/miccai/train_images/`) y guarda sus máscaras correspondientes con el **mismo nombre exacto** en la carpeta de máscaras asociada (por ejemplo, `data/miccai/train_masks/`).
2.  **Actualizar el trackeo en DVC**:
    Ejecuta DVC para recalcular las sumas de verificación (hashes) y registrar las nuevas adiciones:
    ```bash
    # Para el dataset MICCAI:
    dvc add data/miccai
    
    # Para el dataset DFU Tissue:
    dvc add data/dfu_tissue
    ```
3.  **Confirmar los cambios en Git**:
    DVC actualizará automáticamente los archivos `.gitignore` locales y los archivos de puntero `.dvc` (ej. `data/miccai.dvc`). Haz commit de estos punteros en Git para registrar la nueva versión del dataset en el historial del repositorio:
    ```bash
    git add data/miccai.dvc data/dfu_tissue.dvc data/.gitignore
    git commit -m "docs: actualizar dataset de entrenamiento con nuevas imagenes"
    ```
4.  **Descargar los datos en otra máquina**:
    Cualquier otro usuario del repositorio puede sincronizar y descargar esta versión exacta de las imágenes ejecutando:
    ```bash
    git pull
    dvc pull
    ```

### 📁 `docs/`
Contiene la documentación de soporte del repositorio. 
* **`docs/architectures/`**: Alberga los diagramas visuales en formato de imagen (`.png`) correspondientes a las diferentes arquitecturas de redes neuronales implementadas (tales como U-Net, ResUnet, Attention U-Net y U-Net++).

### 📁 `notebooks/`
Espacio de trabajo para la fase de experimentación rápida y visualización de resultados mediante Jupyter Notebooks (`.ipynb`). Aquí se encuentran los cuadernos de pruebas de GPU de PyTorch, pruebas de Double U-Net e inferencia móvil optimizada con TensorFlow Lite.

### 📁 `results/`
Almacena los datos empíricos obtenidos en las evaluaciones de los modelos.
* **`results/classification/`**: Contiene archivos de métricas cuantitativas (archivos `.csv`) asociadas a los algoritmos de clasificación.
* **`results/segmentation/`**: Registra las métricas de rendimiento de las redes de segmentación de DFU y hojas de datos combinadas Excel (`.xlsx`) que resumen los parámetros de los diferentes experimentos.

### 📁 `scripts/`
Contiene los scripts utilitarios principales del proyecto que no forman parte del entrenamiento científico primario:
* **`setup_environment.sh`**: Script para inicializar automáticamente el entorno en `Conda` o `venv`.
* **`check_installation.py`**: Comprobador automático que valida las dependencias y la existencia de aceleración por hardware (CUDA).
* **`track_networks.py`**: Automatización para recopilar los parámetros y métricas de desempeño de las ejecuciones de los modelos.

### 📁 `src/`
El núcleo de la lógica científica del proyecto. Contiene todo el código fuente modular en Python (`.py`):
* **`src/ClasificationAlgorithms/`**: Incluye la lógica de entrenamiento, prueba, carga de datasets de tejido y arquitecturas asociadas a la clasificación de tejidos de úlceras.
* **`src/SegmentationNetworks/`**: Contiene la lógica, sets de imágenes DFU (`data_DFU_images`) y las arquitecturas de segmentación correspondientes.

*Nota: Todos los archivos Python y notebooks dentro de `src/` y `notebooks/` han sido modificados para calcular dinámicamente las rutas de los directorios a través de variables de entorno locales de forma portable, lo que garantiza la ejecución en cualquier sistema sin depender de rutas absolutas locales.*

### 📁 `tests/`
Carpeta reservada para contener suites de pruebas de código a futuro, asegurando que las modificaciones posteriores no rompan las dependencias fundamentales.

### 📁 `weights/`
Carpeta que reúne los archivos finales con los pesos y checkpoints de los modelos entrenados (ej. modelos exportados en formato `.tflite` para ejecución móvil o checkpoints `.pth` de PyTorch).
