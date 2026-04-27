# SLAM Readiness on nuScenes

Este directorio se usa para el experimento limpio y separado de si la pseudo-LiDAR generada desde cámaras es suficientemente útil para tareas tipo SLAM.

Estructura:
- notebooks/: exploración y visualización
- scripts/: automatización reproducible
- manifests/: listas de samples/escenas a evaluar
- outputs/: resultados y métricas
- notes/: apuntes breves de decisiones

Punto 1 actual:
seleccionar varios samples consecutivos de una escena de nuScenes y guardarlos en un manifest para no trabajar sobre un único frame.
