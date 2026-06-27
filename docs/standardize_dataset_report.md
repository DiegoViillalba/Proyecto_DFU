# Standardización del dataset DFUTissue a 256x256 con letterboxing

## Objetivo
Estandarizar físicamente las imágenes y máscaras del dataset local `data/dfu_tissue/` a una resolución uniforme de `256x256` píxeles sin deformar las lesiones ulcerosas.

## Qué hace el script
El nuevo script `scripts/standardize_dataset.py` realiza los siguientes pasos:

1. Itera por las carpetas:
   - `data/dfu_tissue/train_images`
   - `data/dfu_tissue/train_masks`
   - `data/dfu_tissue/val_images`
   - `data/dfu_tissue/val_masks`
   - `data/dfu_tissue/test_images`
   - `data/dfu_tissue/test_masks`
2. Para cada imagen RGB y cada máscara en escala de grises:
   - Redimensiona manteniendo el aspect ratio original.
   - Agrega padding negro (letterbox) hasta `256x256`.
   - Sobrescribe el archivo original con la versión estandarizada.
3. Usa `PIL` para el procesamiento y `tqdm` cuando está disponible para mostrar barras de progreso.
4. Verifica las etiquetas de máscara y reporta warnings si aparecen valores fuera de `{0,1,2,3}`.

## Cómo ejecutar
En tu terminal dentro del repositorio:

```bash
python scripts/standardize_dataset.py
```

Si deseas solamente revisar el estado sin modificar archivos:

```bash
python scripts/standardize_dataset.py --summary-only
```

## Flujo posterior con DVC
Si los archivos en `data/dfu_tissue/` están siendo gestionados por DVC y se modifican físicamente, sí, debemos volver a registrar los cambios con `dvc add`.

### Pasos recomendados tras la estandarización:

```bash
dvc add data/dfu_tissue/train_images
 dvc add data/dfu_tissue/train_masks
 dvc add data/dfu_tissue/val_images
 dvc add data/dfu_tissue/val_masks
 dvc add data/dfu_tissue/test_images
 dvc add data/dfu_tissue/test_masks
```

> Si los archivos estaban ya versionados por DVC, el comando `dvc add` actualizará los punteros en los archivos `.dvc` y generará nuevos hashes para los archivos modificados.

## Interacción con los scripts existentes
- `train_multiseed.py` seguirá funcionando con el mismo dataset, pero ahora los datos estarán físicamente en `256x256`.
- `evaluate_multiseed.py` se beneficiará de la consistencia de tamaño y no necesitará depender de correcciones internas para diferentes resoluciones.
- `scripts/local_data_sanity.py` pasará si los archivos tienen el tamaño esperado y solo usan las etiquetas válidas.

## Siguientes pasos
1. Ejecutar el script de estandarización:
   ```bash
   python scripts/standardize_dataset.py
   ```
2. Comprobar el dataset con el sanity check:
   ```bash
   python scripts/local_data_sanity.py
   ```
3. Registrar cambios en DVC:
   ```bash
   dvc add data/dfu_tissue/train_images
   dvc add data/dfu_tissue/train_masks
   dvc add data/dfu_tissue/val_images
   dvc add data/dfu_tissue/val_masks
   dvc add data/dfu_tissue/test_images
   dvc add data/dfu_tissue/test_masks
   ```
4. Hacer commit de código y documentación:
   ```bash
   git add scripts/standardize_dataset.py docs/standardize_dataset_report.md
   git commit -m "feat: standardize DFUTissue dataset to 256x256 with letterboxing and document DVC workflow"
   ```

## Nota clínica
Usar letterboxing preserva la forma de la lesión original y evita el estiramiento asimétrico del tejido ulceroso, manteniendo el rigor médico requerido para análisis morfológicos posteriores.
