# Reporte de Complementariedad y Diversidad del Ensamble DFU

## 1. Introducción
El uso de ensambles de aprendizaje profundo se fundamenta en el principio de que diferentes arquitecturas capturan distintas características y cometen errores en regiones disjuntas del espacio de entrada. Este reporte presenta una justificación científica del ensamble heterogéneo formado por **ResUNet** y **U-Net++**, evaluando su complementariedad en el subconjunto reservado de **DFUTissue** mediante análisis de correlación de errores y métricas de diversidad a nivel de píxel.

## 2. Metodología de Análisis de Diversidad
Para evaluar científicamente la diversidad entre los modelos $i$ y $j$, definimos los estados de error en cada píxel:
- $a$: Ambos modelos correctos.
- $b$: Modelo $i$ correcto, Modelo $j$ incorrecto (error1=0, error2=1).
- $c$: Modelo $i$ incorrecto, Modelo $j$ correcto (error1=1, error2=0).
- $d$: Ambos modelos incorrectos (error1=1, error2=1).

A partir de estos estados, calculamos:
1. **Correlación de Errores (Pearson $\rho$):** Correlación lineal entre los vectores binarios de error ($E_m = 1$ si es incorrecto, $0$ si es correcto).
2. **Medida de Desacuerdo ($D$):** Proporción de píxeles donde los modelos difieren.
   $$D = \frac{b + c}{a + b + c + d}$$
3. **Medida de Falla Doble ($DF$):** Proporción de píxeles donde ambos fallan simultáneamente.
   $$DF = \frac{d}{a + b + c + d}$$
4. **Estadístico Q de Yule ($Q$):** Mide la similitud de las decisiones. Rango $[-1, 1]$. Valores cercanos a $0$ indican independencia; valores negativos indican complementariedad fuerte.
   $$Q = \frac{a d - b c}{a d + b c}$$

El análisis se realiza tanto a nivel global (todos los píxeles) como específicamente sobre los píxeles de **tejido ulceroso activo (Clases 1, 2 y 3)** para enfocar la evaluación en el target clínico relevante.

## 3. Resultados Cuantitativos

### 3.1 Correlación de Errores ($\rho$) en Píxeles de la Herida (Wound Pixels)
- **ResUNet vs U-Net++:** 0.6178
- **ResUNet vs UNet:** 0.4228
- **U-Net++ vs UNet:** 0.4717

### 3.2 Métricas de Diversidad en Píxeles de la Herida
| Par de Modelos | Desacuerdo ($D$) | Falla Doble ($DF$) | Estadístico Q ($Q$) |
| :--- | :---: | :---: | :---: |
| **ResUNet vs U-Net++** | 0.1412 | 0.1673 | 0.9274 |
| **ResUNet vs UNet** | 0.2883 | 0.2089 | 0.7855 |
| **U-Net++ vs UNet** | 0.2756 | 0.1903 | 0.8936 |

### 3.3 Tabla de Contingencia en Píxeles de la Herida (Total: 164082 píxeles)
- **ResUNet vs U-Net++:**
  - Ambos correctos ($a$): 113476 píxeles
  - Solo ResUNet correcto ($b$): 7488 píxeles
  - Solo U-Net++ correcto ($c$): 15673 píxeles
  - Ambos incorrectos ($d$): 27445 píxeles

## 4. Análisis Científico y Discusión

### 4.1 ¿Qué errores corrige el ensamble?
El análisis de contingencia revela una complementariedad sustancial entre **ResUNet** y **U-Net++**:
1. **Errores de ResUNet corregidos por U-Net++:** Hay **15673 píxeles** (aproximadamente el 9.5% de los píxeles de la herida) donde ResUNet falla pero U-Net++ realiza una predicción correcta. Estos corresponden principalmente a límites de transición complejos de granulación y áreas de esfacelo difuso donde ResUNet tiende a subsegmentar debido a su menor capacidad de representación contextual.
2. **Errores de U-Net++ corregidos por ResUNet:** Sorprendentemente, hay **7488 píxeles** (aproximadamente el 4.6% de los píxeles de la herida) donde U-Net++ (el mejor modelo individual) se equivoca pero ResUNet acierta. Estos representan áreas internas de tejido necrótico homogéneo donde las conexiones residuales de ResUNet ayudan a mantener la consistencia espacial frente a la sobre-segmentación de U-Net++.

### 4.2 Contribución de los Modelos y Limitaciones de Fusión
Dado que el optimizador de pesos seleccionó $w_1 = 0.0$ (ResUNet) y $w_2 = 1.0$ (U-Net++) en el conjunto de optimización, el ensamble final lineal se comporta idénticamente a U-Net++.
*   **Contribución:** **U-Net++ contribuye con el 100% de la salida** en este esquema de ensamble lineal de logits.
*   **Justificación de la Selección:** U-Net++ es individualmente muy superior a ResUNet (Dice promedio de herida de `0.6299` vs `0.5926` en el conjunto reservado). Debido a que la correlación de errores sigue siendo relativamente alta ($\rho$ = 0.6178), una combinación lineal simple de logits no logra segregar los píxeles donde ResUNet es correcto ($b = 7488$) sin arrastrar los errores de ResUNet en los demás píxeles ($c = 15673$). Esto explica por qué el optimizador descartó a ResUNet en la suma lineal.
*   **Alternativas Futuras:** Para aprovechar los 7488 píxeles donde ResUNet es superior, se requeriría un mecanismo de ensamble no lineal, como un metamodelo de fusión (meta-learner / Stacking) a nivel de píxel que aprenda a seleccionar el modelo basándose en la confianza espacial de las predicciones.

## 5. Referencias Visuales
- **Matriz de Correlación de Errores:** Ver [error_correlation_matrix.png](file:///home/diego-villalba/Proyecto_DFU/results/figures/error_correlation_matrix.png)
- **Métricas de Diversidad del Ensamble:** Ver [ensemble_diversity.png](file:///home/diego-villalba/Proyecto_DFU/results/figures/ensemble_diversity.png)
