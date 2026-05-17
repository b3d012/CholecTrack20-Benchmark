# EfficientDet-D0 Paper-Style Detection Results

Command:

```powershell
python -m src.train paper-detection-eval --weights results/logs/efficientdet_d0_640_adamw_lr2e4_long/weights/best.pth --split val --half --model-name EfficientDet-D0-640-AdamW-lr2e4
```

Selected checkpoint:

```text
results/logs/efficientdet_d0_640_adamw_lr2e4_long/weights/best.pth
```

| Detection model | AP0.5 | AP0.75 | AP0.5:0.95 | Grasper | Bipolar | Hook | Scissors | Clipper | Irrigator | Bag | Bleeding | Blur | Smoke | Crowded | Occluded | Reflection | Foul Lens | Trocar | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientDet-D0-640-AdamW-lr2e4 | 44.4 | 31.2 | 27.9 | 47.3 | 50.8 | 60.3 | 33.4 | 60.5 | 0.6 | 57.6 | 19.2 | 75.2 | 46.9 | 24.4 | 52.0 |  | 17.1 | 47.3 | 20.9 |

The weakest class is `irrigator`, which suggests the main detector limitation is data/class imbalance or difficult class appearance, not only optimizer choice.

