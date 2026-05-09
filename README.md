# Evaluación de pseudo-LiDAR para registro temporal tipo SLAM en nuScenes

Este repositorio contiene el código experimental desarrollado para estudiar si una pseudo-LiDAR generada a partir de cámaras RGB multicámara puede ser útil para tareas de registro temporal tipo SLAM.

El objetivo no es únicamente generar una nube de puntos visualmente razonable, sino comprobar si esa nube mantiene suficiente consistencia geométrica entre frames consecutivos como para estimar movimiento relativo.

## 1. Pregunta experimental

La pregunta principal del trabajo es:

> ¿La pseudo-LiDAR generada desde las cámaras de nuScenes contiene información geométrica suficientemente estable para tareas de registro temporal similares a las empleadas en LiDAR-SLAM?

Para responder a esta pregunta se evalúan dos aspectos:

- la calidad geométrica de la pseudo-LiDAR frente al LiDAR real `LIDAR_TOP`
- la estabilidad temporal de la pseudo-LiDAR al registrar frames consecutivos

## 2. Contexto del experimento

La pipeline general es:

```text
6 cámaras RGB
-> Depth Pro
-> mapas de profundidad
-> reproyección 3D
-> fusión en ego frame
-> pseudo-LiDAR
-> comparación con LIDAR_TOP
-> registro temporal
-> análisis de utilidad para SLAM
```

El experimento parte de una pseudo-LiDAR 360 generada con las seis cámaras de nuScenes. Después se estudia si utilizar todo el anillo es realmente lo más adecuado para registro temporal o si algunas regiones son más estables que otras.

## 3. Requisitos externos

Este repositorio no incluye ni el dataset ni el repositorio de Depth Pro. Para reproducir el trabajo se necesita:

- una instalación funcional de **Depth Pro**
- el checkpoint de Depth Pro
- el dataset **nuScenes**, por ejemplo `v1.0-mini`
- un entorno Python con las dependencias necesarias

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
export NUSCENES_ROOT=/ruta/al/dataset/nuscenes
```

`DEPTH_PRO_ROOT` debe apuntar al repositorio de Depth Pro.

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
- anillo pseudo-LiDAR completo frente a la región frontal selecconada
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
- `notebooks/02_generate_pseudolidar_from_manifest.ipynb`: generación de pseudo-LiDAR.
- `notebooks/03_pairwise_registration.ipynb`: primer análisis de registro entre frames.
- `notebooks/06_region_analysis.ipynb`: análisis por regiones y distancia.
- `notebooks/07_subset_registration_comparison.ipynb`: comparación entre anillo completo y subconjuntos.
- `notebooks/08_front_registration_strategies.ipynb`: estrategias de registro frontal.
- `notebooks/10_front_tuning_experiments.ipynb`: ajuste fino de selección y parámetros.
- `notebooks/11_accumulated_trajectory.ipynb`: trayectoria acumulada.
- `notebooks/12_pointcloud_comparison_best_config.ipynb`: comparación visual de nubes.
- `notebooks/13_final_conclusions_and_results.ipynb`: conclusiones finales.

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

### Error acumulado

Mide cómo crece el error al encadenar transformaciones entre varios frames consecutivos. Es importante porque se aproxima más a una situación tipo SLAM u odometría.

## 9. Resultados principales

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

## 11. Outputs principales

Los resultados principales se generan en:

```text
outputs/scene-0061
```

Ficheros JSON importantes:

- `outputs/scene-0061/run_summary.json`
- `outputs/scene-0061/pairwise_registration_metrics.json`
- `outputs/scene-0061/pairwise_region_analysis.json`
- `outputs/scene-0061/subset_registration_comparison.json`
- `outputs/scene-0061/front_registration_strategies.json`
- `outputs/scene-0061/front_point_selection_comparison.json`
- `outputs/scene-0061/front_global_icp_param_sweep.json`
- `outputs/scene-0061/front_best_combination.json`
- `outputs/scene-0061/accumulated_trajectory_comparison.json`

Figuras importantes:

- `outputs/scene-0061/accumulated_trajectory_xy.png`
- `outputs/scene-0061/accumulated_trajectory_drift.png`
- `outputs/scene-0061/pointcloud_comparison_best_cfg05.png`
- `outputs/scene-0061/pointcloud_comparison_narrow_cfg05.png`

Visualizaciones 3D interactivas:

- `outputs/scene-0061/pointcloud_phase_visualizations/index.html`
- `outputs/scene-0061/pointcloud_phase_visualizations/phase_01_dense_ring_6cams.html`
- `outputs/scene-0061/pointcloud_phase_visualizations/phase_02_pseudolidar_vs_lidar_gt.html`
- `outputs/scene-0061/pointcloud_phase_visualizations/phase_03_ring_pseudo_lidar_overlay.html`
- `outputs/scene-0061/pointcloud_phase_visualizations/phase_03b_full_ring_vs_front_subset.html`
- `outputs/scene-0061/pointcloud_phase_visualizations/phase_03c_full_ring_after_icp.html`
- `outputs/scene-0061/pointcloud_phase_visualizations/phase_04a_front_pair_before_registration.html`
- `outputs/scene-0061/pointcloud_phase_visualizations/phase_04b_front_pair_after_cfg05.html`

## 12. Conclusión experimental

La pseudo-LiDAR generada desde cámaras no sustituye directamente a un LiDAR real para SLAM robusto. Sin embargo, sí contiene información geométrica útil para registro temporal si se utiliza de forma selectiva.

La conclusión principal es que la pseudo-LiDAR no tiene la misma fiabilidad en todas las zonas. En este experimento, la parte frontal funciona mejor que usar todo el anillo y combinar una alineación global con ICP ayuda a estimar mejor el movimiento entre frames.

Por tanto, el resultado no demuestra una sustitución completa del LiDAR, pero sí una utilidad real de la pseudo-LiDAR como representación intermedia para análisis geométrico y registro temporal.
