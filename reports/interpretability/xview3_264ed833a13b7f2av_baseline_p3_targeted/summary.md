## D4 con ground truth — escena `264ed833a13b7f2av`, conjunto `aidra` (20 muestras)

| Estrato | Pool | Muestras | Grad-CAM pointing-game | Grad-CAM masa en caja | CFAR pointing-game | CFAR masa en caja |
|---|---:|---:|---:|---:|---:|---:|
| `tp_high` | 18 | 5 | 0.6 | 0.567 | 0.8 | 0.294 |
| `tp_low` | 17 | 5 | 0.2 | 0.149 | 0.2 | 0.071 |
| `fp` | 106 | 5 | 0.4 | 0.318 | 0.6 | 0.249 |
| `fn` | 12 | 5 | 0.2 | 0.171 | 0.8 | 0.036 |

Grad-CAM `targeted` @ `model.model.15`; renderer: `vesseltracker-sar-yolov8` (`18aec1bb3caf…`); sujeto: `vesseltracker-sar-yolov8` (`18aec1bb3caf…`); CFAR guard/training 8/20; tolerancia 20 px; seed 42; commit `59e028b4667d`.
