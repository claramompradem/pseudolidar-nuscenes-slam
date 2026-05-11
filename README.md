# Evaluación de pseudo-LiDAR para registro temporal tipo SLAM en nuScenes

Este repositorio contiene el código experimental desarrollado para estudiar si una pseudo-LiDAR generada a partir de cámaras RGB multicámara puede ser útil para tareas de registro temporal tipo SLAM.

El objetivo no es únicamente generar una nube de puntos visualmente razonable, sino comprobar si esa nube mantiene suficiente consistencia geométrica entre frames consecutivos como para estimar movimiento relativo.

## 1. Pregunta experimental

La pregunta principal del trabajo es:

> ¿La pseudo-LiDAR generada desde las cámaras de nuScenes contiene información geométrica suficientemente estable para tareas de registro temporal similares a las empleadas en LiDAR-SLAM?

Para responder a esta pregunta se evalúan dos aspectos:

- la calidad geométrica de la pseudo-LiDAR frente al LiDAR real `LIDAR_TOP`
- la estabilidad temporal de la pseudo-LiDAR al registrar frames consecutivos
- la diferencia entre registrar pseudo-LiDAR y registrar LiDAR real con el mismo backend

## 2. Contexto del experimento

La pipeline general es:

```text
6 cámaras RGB
-> comparación previa Depth Pro / Depth Anything 3
-> Depth Pro como modelo base
-> mapas de profundidad
-> reproyección 3D
-> fusión en ego frame
-> pseudo-LiDAR
-> comparación con LIDAR_TOP
-> consistencia temporal con pose real
-> registro temporal
-> baseline de registro con LiDAR real
-> análisis de utilidad para SLAM
```

El experimento parte de una comparación inicial entre `Depth Pro` y `Depth Anything 3` para justificar el modelo de profundidad usado como baseline. A partir de esa decisión, se genera una pseudo-LiDAR 360 con las seis cámaras de nuScenes. Después se estudia si utilizar todo el anillo es realmente lo más adecuado para registro temporal o si algunas regiones son más estables que otras.

## 3. Requisitos externos

Este repositorio no incluye ni el dataset ni el repositorio de Depth Pro. Para reproducir el trabajo se necesita:

- una instalación funcional de **Depth Pro**
- el checkpoint de Depth Pro
- una instalación funcional de **Depth Anything 3** si se quiere reproducir también la comparación entre modelos de profundidad desde inferencia
- el dataset **nuScenes**, por ejemplo `v1.0-mini`
- un entorno Python con las dependencias necesarias

`Depth Anything 3` se utiliza como comparación auxiliar de calidad geométrica. No es necesario para ejecutar la rama principal de registro temporal, pero sí para reproducir desde cero la comparación entre modelos de profundidad.

Dependencias principales:

- `torch`
- `numpy`
- `open3d`
- `Pillow`
- `pyquaternion`
- `nuscenes-devkit`
- `depth_pro`

La forma más sencilla de reproducir el trabajo es usar el mismo entorno en el que esté instalado Depth Pro y añadir este repositorio como carpeta de experimentos.

## 4. Variables recomendadas

Para que los comandos sean portables, se recomienda definir dos rutas:

```bash
export DEPTH_PRO_ROOT=/ruta/al/repositorio/ml-depth-pro
export DEPTH_ANYTHING3_ROOT=/ruta/al/repositorio/depth-anything-3
export NUSCENES_ROOT=/ruta/al/dataset/nuscenes
```

`DEPTH_PRO_ROOT` debe apuntar al repositorio de Depth Pro.

`DEPTH_ANYTHING3_ROOT` debe apuntar al repositorio de Depth Anything 3. Si el paquete `depth_anything_3` ya está instalado en el entorno, esta variable puede omitirse.

`NUSCENES_ROOT` debe apuntar a la carpeta donde está el dataset nuScenes.

En Windows con WSL se pueden definir igual dentro de la terminal de Ubuntu.

## 5. Estructura del repositorio

```text
slam_readiness_nuscenes/
|-- manifests/
|-- notebooks/
|-- outputs/
|-- scripts/
`-- README.md
```

### `manifests`

Contiene ficheros JSON con la selección de escenas y samples utilizados.

Ejemplo:

```text
manifests/scene-0061_first5.json
```

Este manifiesto fija los primeros cinco samples consecutivos de la escena `scene-0061`.

### `scripts`

Contiene los scripts reproducibles del experimento. Son la parte más importante si se quiere ejecutar el flujo completo desde código.

### `notebooks`

Contiene notebooks de análisis, visualización e interpretación de resultados.

### `outputs`

Contiene los resultados generados:

- nubes pseudo-LiDAR por frame
- nubes densas fusionadas
- métricas en formato JSON
- comparación auxiliar entre modelos de profundidad
- figuras PNG
- comparaciones de trayectoria
- visualizaciones de nubes de puntos

## 6. Scripts principales

### `scripts/build_scene_manifest.py`

Construye un manifiesto con varios samples consecutivos de una escena de nuScenes.

Su función es evitar trabajar sobre un único frame aislado y preparar una secuencia temporal reproducible.

Ejemplo:

```bash
python scripts/build_scene_manifest.py \
  --dataroot "$NUSCENES_ROOT" \
  --version v1.0-mini \
  --scene-name scene-0061 \
  --num-samples 5 \
  --output manifests/scene-0061_first5.json
```

### `scripts/generate_pseudolidar_manifest.py`

Genera la pseudo-LiDAR para todos los samples incluidos en el manifiesto.

Para cada frame:

- carga las seis cámaras de nuScenes
- estima profundidad con Depth Pro
- reproyecta cada mapa de profundidad a 3D
- transforma cada nube al frame del vehículo
- fusiona las seis vistas
- genera la pseudo-LiDAR
- guarda los resultados en `outputs`

Ejemplo:

```bash
python scripts/generate_pseudolidar_manifest.py \
  --manifest manifests/scene-0061_first5.json \
  --output-root outputs \
  --depth-pro-root "$DEPTH_PRO_ROOT"
```

El argumento `--depth-pro-root` permite usar cualquier instalación local de Depth Pro (puede omitirse si se ha definido la variable de entorno `DEPTH_PRO_ROOT`).

### `scripts/generate_depthanything3_sample.py`

Ejecuta la inferencia de `Depth Anything 3` sobre un sample de nuScenes y genera una pseudo-LiDAR equivalente a la de `Depth Pro`.

Este script se utiliza para que la comparación entre modelos de profundidad sea reproducible. No parte de una nube externa ya preparada, sino que:

- carga las seis cámaras del sample seleccionado
- ejecuta `Depth Anything 3`
- guarda los mapas de profundidad por cámara
- reconstruye la nube densa fusionada en ego frame
- genera la pseudo-LiDAR
- guarda los `.ply` resultantes dentro de `outputs`

Ejemplo:

```bash
python scripts/generate_depthanything3_sample.py \
  --manifest manifests/scene-0061_first5.json \
  --output-root outputs \
  --sample-index 0 \
  --depth-anything-root "$DEPTH_ANYTHING3_ROOT"
```

Si `depth_anything_3` ya está instalado en el entorno Python, el argumento `--depth-anything-root` puede omitirse.

### `scripts/compare_depth_models_common_support.py`

Recalcula la comparación entre la pseudo-LiDAR generada con `Depth Pro` y la pseudo-LiDAR generada con `Depth Anything 3`.

Este script no copia métricas guardadas previamente: carga las nubes `.ply`, filtra ambas al mismo soporte común y calcula de nuevo las distancias nearest-neighbor frente a `LIDAR_TOP` y la IoU de ocupación BEV.

Ejemplo:

```bash
python scripts/compare_depth_models_common_support.py \
  --run-summary outputs/scene-0061/run_summary.json \
  --depth-anything-sample-dir outputs/scene-0061/depthanything3_sample_000 \
  --sample-index 0 \
  --output outputs/scene-0061/depth_model_comparison_common_support.json
```

La carpeta indicada en `--depth-anything-sample-dir` debe contener, como mínimo, el archivo:

```text
pcd_pseudolidar_ego.ply
```

En este repositorio se incluye el recurso mínimo necesario para reproducir la comparación ya evaluada:

```text
outputs/scene-0061/depthanything3_sample_000/pcd_pseudolidar_ego.ply
```

### `scripts/evaluate_pairwise_registration.py`

Evalúa el registro entre frames consecutivos.

Compara el movimiento estimado al registrar la pseudo-LiDAR del frame `t` contra la pseudo-LiDAR del frame `t+1` con el movimiento real obtenido de las poses de nuScenes.

Ejemplo:

```bash
python scripts/evaluate_pairwise_registration.py \
  --run-summary outputs/scene-0061/run_summary.json \
  --dataroot "$NUSCENES_ROOT" \
  --version v1.0-mini
```

### `scripts/evaluate_lidar_registration_baseline.py`

Evalúa el registro temporal usando el LiDAR real `LIDAR_TOP` como entrada.

La finalidad es obtener un baseline con geometría métrica real usando un backend comparable al de pseudo-LiDAR. Si el LiDAR real registra mucho mejor que la pseudo-LiDAR, parte de la diferencia se debe a la representación generada desde cámaras y no solo al algoritmo ICP.

El script:

- lee `outputs/scene-0061/run_summary.json`
- carga las nubes `LIDAR_TOP` guardadas por sample
- registra pares consecutivos con ICP directo
- prueba también una variante de alineación global seguida de ICP
- compara la transformación estimada con la pose real de nuScenes
- genera JSON, CSV y una figura comparativa con pseudo-LiDAR si existen los JSON previos

Ejemplo:

```bash
python scripts/evaluate_lidar_registration_baseline.py \
  --run-summary outputs/scene-0061/run_summary.json \
  --dataroot "$NUSCENES_ROOT" \
  --version v1.0-mini \
  --output outputs/scene-0061/lidar_registration_baseline.json
```

Genera:

- `outputs/scene-0061/lidar_registration_baseline.json`
- `outputs/scene-0061/lidar_registration_baseline.csv`
- `outputs/scene-0061/lidar_vs_pseudolidar_registration_baseline.png`

### `scripts/evaluate_temporal_consistency.py`

Evalúa la consistencia temporal de la pseudo-LiDAR sin estimar la pose con ICP.

Para cada par de frames consecutivos:

- carga la pseudo-LiDAR del frame `t`
- carga la pseudo-LiDAR del frame `t+1`
- obtiene la transformación real entre ambos frames a partir de las poses de nuScenes
- transforma la nube de `t` al sistema de coordenadas de `t+1`
- compara ambas nubes mediante nearest-neighbor y ratios de solape

Este análisis permite separar dos fuentes de error:

- errores propios de la representación pseudo-LiDAR
- errores introducidos posteriormente por el algoritmo de registro

Ejemplo:

```bash
python scripts/evaluate_temporal_consistency.py \
  --run-summary outputs/scene-0061/run_summary.json \
  --dataroot "$NUSCENES_ROOT" \
  --version v1.0-mini \
  --output outputs/scene-0061/temporal_consistency_metrics.json
```

Genera:

- `outputs/scene-0061/temporal_consistency_metrics.json`
- `outputs/scene-0061/temporal_consistency_by_region.csv`
- `outputs/scene-0061/temporal_consistency_by_region.png`

### `scripts/analyze_pairwise_regions.py`

Analiza el registro separando la nube por regiones espaciales y por distancia.

Regiones evaluadas:

- `all`
- `near`
- `mid`
- `far`
- `front`
- `left`
- `right`

Ejemplo:

```bash
python scripts/analyze_pairwise_regions.py \
  --run-summary outputs/scene-0061/run_summary.json \
  --dataroot "$NUSCENES_ROOT" \
  --version v1.0-mini
```

### `scripts/compare_registration_subsets.py`

Compara el registro usando distintos subconjuntos espaciales:

- anillo completo (`all`)
- región frontal (`front`)
- frontal izquierda (`front_left`)
- frontal derecha (`front_right`)

Ejemplo:

```bash
python scripts/compare_registration_subsets.py \
  --run-summary outputs/scene-0061/run_summary.json \
  --dataroot "$NUSCENES_ROOT" \
  --version v1.0-mini
```

### `scripts/compare_front_registration_strategies.py`

Compara dos estrategias sobre la región frontal:

- ICP directo
- alineación global seguida de ICP

Ejemplo:

```bash
python scripts/compare_front_registration_strategies.py \
  --run-summary outputs/scene-0061/run_summary.json \
  --dataroot "$NUSCENES_ROOT" \
  --version v1.0-mini
```

### `scripts/compare_front_point_selection_strategies.py`

Evalúa distintas formas de seleccionar puntos dentro de la región frontal.

Variantes principales:

- `front_baseline`
- `front_narrow`
- `front_compact`
- `front_midrange`
- `front_above_ground`

### `scripts/sweep_front_global_icp_params.py`

Realiza un barrido de parámetros para la estrategia de alineación global seguida de ICP.

Se ajustan parámetros como:

- tamaño de voxel para la fase gruesa
- umbral de correspondencia global
- tamaño de voxel para la fase fina
- umbral de correspondencia de ICP

La mejor configuración encontrada es `cfg_05`.

### `scripts/evaluate_front_best_combination.py`

Evalúa las mejores combinaciones encontradas en los experimentos anteriores.

Compara principalmente:

- `front_baseline + cfg_05`
- `front_narrow + cfg_05`

### `scripts/evaluate_accumulated_trajectory.py`

Evalúa la trayectoria acumulada al encadenar transformaciones entre varios frames consecutivos.

Este análisis es importante para SLAM porque un método puede funcionar bien entre dos frames concretos pero acumular error cuando se encadenan varios movimientos.

Ejemplo:

```bash
python scripts/evaluate_accumulated_trajectory.py \
  --run-summary outputs/scene-0061/run_summary.json \
  --dataroot "$NUSCENES_ROOT" \
  --version v1.0-mini
```

### `scripts/create_pointcloud_phase_visualizations.py`

Genera visualizaciones 3D interactivas en HTML para inspeccionar distintas fases de la nube de puntos.

Las fases visualizadas son:

- nube densa fusionada desde las seis cámaras
- pseudo-LiDAR frente al LiDAR real
- superposición de nube densa, pseudo-LiDAR y LiDAR real
- anillo pseudo-LiDAR completo frente a la región frontal seleccionada
- registro usando el anillo completo
- región frontal de dos frames consecutivos antes del registro
- región frontal de esos mismos frames después de aplicar `cfg_05`

Ejemplo:

```bash
python scripts/create_pointcloud_phase_visualizations.py \
  --run-summary outputs/scene-0061/run_summary.json
```

Los HTML se generan en:

```text
outputs/scene-0061/pointcloud_phase_visualizations
```

## 7. Notebooks

Los notebooks están en:

```text
notebooks/
```

Los más importantes son:

- `notebooks/01_select_samples_nuscenes.ipynb`: selección de escena y samples.
- `notebooks/02_generate_pseudolidar_from_manifest.ipynb`: generación con Depth Pro, inferencia de Depth Anything 3 para el sample de comparación y evaluación geométrica entre ambos.
- `notebooks/03_pairwise_registration.ipynb`: primer análisis de registro entre frames.
- `notebooks/06_region_analysis.ipynb`: análisis por regiones y distancia.
- `notebooks/07_subset_registration_comparison.ipynb`: comparación entre anillo completo y subconjuntos.
- `notebooks/08_front_registration_strategies.ipynb`: estrategias de registro frontal.
- `notebooks/10_front_tuning_experiments.ipynb`: ajuste fino de selección y parámetros.
- `notebooks/11_accumulated_trajectory.ipynb`: trayectoria acumulada.
- `notebooks/12_pointcloud_comparison_best_config.ipynb`: comparación visual de nubes.
- `notebooks/13_temporal_consistency_analysis.ipynb`: consistencia temporal usando la pose real de nuScenes.
- `notebooks/14_lidar_baseline_comparison.ipynb`: comparación entre registro con LiDAR real y registro con pseudo-LiDAR.
- `notebooks/15_ablation_summary.ipynb`: resumen de ablaciones experimentales.
- `notebooks/16_final_experimental_synthesis.ipynb`: síntesis final de resultados y conclusiones.

## 8. Métricas utilizadas

### Error de traslación

Mide la diferencia, en metros, entre la traslación estimada por el registro y la traslación real obtenida de las poses de nuScenes.

### Error de rotación

Mide la diferencia angular, en grados, entre la rotación estimada y la rotación real.

### Fitness de ICP

Indica qué proporción de puntos encuentra correspondencias válidas durante ICP.

Un valor alto de fitness no garantiza por sí solo una buena estimación del movimiento. Puede haber muchas correspondencias, pero estar asociadas a una transformación incorrecta.

### RMSE

Mide el error medio de las correspondencias aceptadas durante ICP.

### Baseline con LiDAR real

Mide cuánto error obtiene el mismo backend de registro cuando la entrada es el LiDAR real `LIDAR_TOP`.

Este baseline sirve como referencia de geometría fiable. No representa una solución cámara-only, sino un límite superior práctico para interpretar cuánto se aleja la pseudo-LiDAR del comportamiento del sensor LiDAR real.

### Consistencia temporal con pose real

Mide si la pseudo-LiDAR generada en frames consecutivos representa de forma parecida la misma escena cuando se alinean las nubes usando la pose real de nuScenes.

Las métricas usadas son:

- distancia nearest-neighbor media
- distancia nearest-neighbor mediana
- percentil 90 de distancia nearest-neighbor
- ratio de solape a `0.5 m`
- ratio de solape a `1.0 m`

Esta evaluación no mide la calidad de ICP, sino la estabilidad temporal de la representación pseudo-LiDAR.

### Error acumulado

Mide cómo crece el error al encadenar transformaciones entre varios frames consecutivos. Es importante porque se aproxima más a una situación tipo SLAM u odometría.

## 9. Resultados principales

### Comparación entre modelos de profundidad

Antes de fijar la rama principal del experimento, se compara la pseudo-LiDAR obtenida con `Depth Pro` y con `Depth Anything 3` en soporte común frente al LiDAR real.

Resultados aproximados:

- `Depth Pro`: mediana pseudo-LiDAR -> GT de `0.52 m` e IoU BEV de `0.236`
- `Depth Anything 3`: mediana pseudo-LiDAR -> GT de `0.63 m` e IoU BEV de `0.227`

En este experimento, `Depth Pro` ofrece una geometría algo más consistente, por lo que se mantiene como modelo base para el análisis de registro temporal.

### Anillo completo frente a región frontal

El uso de todo el anillo 360 no produce el mejor registro temporal.

Resultados aproximados:

- anillo completo: `2.18 m` de error medio de traslación
- región frontal: `1.91 m` de error medio de traslación

Esto indica que la pseudo-LiDAR no tiene una fiabilidad homogénea en todas las direcciones.

### Alineación global seguida de ICP

Sobre la región frontal, añadir una fase de alineación global antes de ICP mejora el resultado.

Resultados aproximados:

- `front + ICP directo`: `1.91 m` y `2.33 grados`
- `front + alineación global + ICP`: `1.30 m` y `1.80 grados`

### Ajuste fino de parámetros

El barrido de parámetros reduce el error hasta aproximadamente:

- `1.18 m` de error medio de traslación
- `1.60 grados` de error medio de rotación

### Consistencia temporal

Al alinear pseudo-LiDARs consecutivas usando la pose real de nuScenes, la región frontal presenta mejor coherencia temporal que el anillo completo.

Resultados aproximados:

- anillo completo: `0.91 m` de distancia nearest-neighbor media y `0.73` de solape a `1.0 m`
- región frontal: `0.71 m` de distancia nearest-neighbor media y `0.79` de solape a `1.0 m`
- región cercana: `0.49 m` de distancia nearest-neighbor media y `0.88` de solape a `1.0 m`
- región lejana: `1.86 m` de distancia nearest-neighbor media y `0.49` de solape a `1.0 m`

Esto ayuda a explicar por qué las regiones cercanas o frontales son más útiles para registro que zonas lejanas o menos estables.

### Baseline LiDAR real

Al registrar `LIDAR_TOP` real con el mismo backend de ICP, el error con la nube completa es mucho menor que con pseudo-LiDAR.

Resultados aproximados:

- `LiDAR real all + ICP`: `0.07 m` y `0.22 grados`
- `pseudo-LiDAR all + ICP`: `1.51 m` y `1.27 grados`
- `pseudo-LiDAR front cfg_05`: `1.18 m` y `1.60 grados`

Esto confirma que el backend de registro puede funcionar bien cuando la geometría es fiable, y que una parte importante del error de pseudo-LiDAR procede de la representación generada desde cámaras.

### Trayectoria acumulada

La evaluación de trayectoria acumulada muestra que la configuración localmente más agresiva no siempre es la más estable al encadenar varios frames.

La configuración más equilibrada para una lectura tipo SLAM es:

- región frontal base
- alineación global seguida de ICP
- configuración `cfg_05`

## 10. Cómo reproducir el experimento

Se asume que este repositorio se ha descargado de forma independiente y que se ejecutan los comandos desde su raíz:

```bash
cd slam_readiness_nuscenes
```

Definir rutas externas:

```bash
export DEPTH_PRO_ROOT=/ruta/al/repositorio/ml-depth-pro
export NUSCENES_ROOT=/ruta/al/dataset/nuscenes
```

Crear o reutilizar el manifiesto:

```bash
python scripts/build_scene_manifest.py \
  --dataroot "$NUSCENES_ROOT" \
  --version v1.0-mini \
  --scene-name scene-0061 \
  --num-samples 5 \
  --output manifests/scene-0061_first5.json
```

Generar la pseudo-LiDAR:

```bash
python scripts/generate_pseudolidar_manifest.py \
  --manifest manifests/scene-0061_first5.json \
  --output-root outputs \
  --depth-pro-root "$DEPTH_PRO_ROOT"
```

Evaluar el registro entre frames:

```bash
python scripts/evaluate_pairwise_registration.py \
  --run-summary outputs/scene-0061/run_summary.json \
  --dataroot "$NUSCENES_ROOT" \
  --version v1.0-mini
```

Analizar regiones:

```bash
python scripts/analyze_pairwise_regions.py \
  --run-summary outputs/scene-0061/run_summary.json \
  --dataroot "$NUSCENES_ROOT" \
  --version v1.0-mini
```

Comparar subconjuntos:

```bash
python scripts/compare_registration_subsets.py \
  --run-summary outputs/scene-0061/run_summary.json \
  --dataroot "$NUSCENES_ROOT" \
  --version v1.0-mini
```

Evaluar trayectoria acumulada:

```bash
python scripts/evaluate_accumulated_trajectory.py \
  --run-summary outputs/scene-0061/run_summary.json \
  --dataroot "$NUSCENES_ROOT" \
  --version v1.0-mini
```

Evaluar consistencia temporal:

```bash
python scripts/evaluate_temporal_consistency.py \
  --run-summary outputs/scene-0061/run_summary.json \
  --dataroot "$NUSCENES_ROOT" \
  --version v1.0-mini \
  --output outputs/scene-0061/temporal_consistency_metrics.json
```

Evaluar baseline LiDAR real:

```bash
python scripts/evaluate_lidar_registration_baseline.py \
  --run-summary outputs/scene-0061/run_summary.json \
  --dataroot "$NUSCENES_ROOT" \
  --version v1.0-mini \
  --output outputs/scene-0061/lidar_registration_baseline.json
```

## 11. Outputs principales

Los resultados principales se generan en:

```text
outputs/scene-0061
```

Ficheros JSON importantes:

- `outputs/scene-0061/run_summary.json`
- `outputs/scene-0061/depth_model_comparison_common_support.json`
- `outputs/scene-0061/pairwise_registration_metrics.json`
- `outputs/scene-0061/pairwise_region_analysis.json`
- `outputs/scene-0061/subset_registration_comparison.json`
- `outputs/scene-0061/front_registration_strategies.json`
- `outputs/scene-0061/front_point_selection_comparison.json`
- `outputs/scene-0061/front_global_icp_param_sweep.json`
- `outputs/scene-0061/front_best_combination.json`
- `outputs/scene-0061/accumulated_trajectory_comparison.json`
- `outputs/scene-0061/temporal_consistency_metrics.json`
- `outputs/scene-0061/temporal_consistency_by_region.csv`
- `outputs/scene-0061/lidar_registration_baseline.json`
- `outputs/scene-0061/lidar_registration_baseline.csv`

Recurso auxiliar para comparar modelos de profundidad:

- `outputs/scene-0061/depthanything3_sample_000/pcd_pseudolidar_ego.ply`
- `outputs/scene-0061/depthanything3_sample_000/summary.json`

Figuras importantes:

- `outputs/scene-0061/accumulated_trajectory_xy.png`
- `outputs/scene-0061/accumulated_trajectory_drift.png`
- `outputs/scene-0061/temporal_consistency_by_region.png`
- `outputs/scene-0061/lidar_vs_pseudolidar_registration_baseline.png`
- `outputs/scene-0061/pointcloud_comparison_best_cfg05.png`
- `outputs/scene-0061/pointcloud_comparison_narrow_cfg05.png`

Visualizaciones 3D interactivas:

- `outputs/scene-0061/pointcloud_phase_visualizations/index.html`
- `outputs/scene-0061/pointcloud_phase_visualizations/phase_01_dense_ring_6cams.html`
- `outputs/scene-0061/pointcloud_phase_visualizations/phase_02_pseudolidar_vs_lidar_gt.html`
- `outputs/scene-0061/pointcloud_phase_visualizations/phase_03_ring_pseudo_lidar_overlay.html`
- `outputs/scene-0061/pointcloud_phase_visualizations/phase_03a_depthpro_depthanything3_lidar.html`
- `outputs/scene-0061/pointcloud_phase_visualizations/phase_03b_full_ring_vs_front_subset.html`
- `outputs/scene-0061/pointcloud_phase_visualizations/phase_03c_full_ring_after_icp.html`
- `outputs/scene-0061/pointcloud_phase_visualizations/phase_04a_front_pair_before_registration.html`
- `outputs/scene-0061/pointcloud_phase_visualizations/phase_04b_front_pair_after_cfg05.html`

## 12. Conclusión experimental

La pseudo-LiDAR generada desde cámaras no sustituye directamente a un LiDAR real para SLAM robusto. Sin embargo, sí contiene información geométrica útil para registro temporal si se utiliza de forma selectiva.

La conclusión principal es que la pseudo-LiDAR no tiene la misma fiabilidad en todas las zonas. En este experimento, la parte frontal y las zonas cercanas presentan mejor consistencia temporal que las zonas lejanas, y la parte frontal funciona mejor que usar todo el anillo para el registro.

Por tanto, el resultado no demuestra una sustitución completa del LiDAR, pero sí una utilidad real de la pseudo-LiDAR como representación intermedia para análisis geométrico, consistencia temporal y registro entre frames.
