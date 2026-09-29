| Dataset | Detector | Kendall tau, FragA vs negated CLEAR Frag per track | Kendall tau, FragA vs negated IDSW per track | same, PoseTrack21 FragA | Kendall tau, FA-HOTA vs HOTA | same, PoseTrack21 FA-HOTA |
|---|---|---|---|---|---|---|
| MOT17 train | DPM | 0.64 | 0.50 | 0.50 | 0.79 | 0.93 |
| MOT17 train | FRCNN | 0.71 | 0.07 | 0.21 | 0.93 | 1.00 |
| MOT17 train | SDP | 0.79 | 0.36 | 0.29 | 0.71 | 1.00 |
| MOT17 train | YOLOX | 0.71 | 0.43 | 0.57 | 0.36 | 1.00 |
| DanceTrack val | YOLOX | 0.86 | 0.36 | 0.79 | 0.43 | 0.50 |

| Dataset | Spearman, FragA vs CLEAR Frag per track | same, PoseTrack21 | Spearman, FragA vs IDSW per track | same, PoseTrack21 | Spearman, FA-HOTA vs HOTA (pooled) | same, PoseTrack21 (pooled) |
|---|---|---|---|---|---|---|
| MOT17 train | -0.54 | -0.30 | -0.23 | -0.14 | 0.9677 | 0.9996 |
| DanceTrack val | -0.93 | -0.71 | -0.57 | -0.90 | 0.6905 | 0.7381 |
