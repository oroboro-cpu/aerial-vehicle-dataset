# QA notes: label review and pipeline log

Side notes that did not fit into the README. Numbers come from `results/label_edits.csv`.

## Error types seen during review

| Error type | Where it was most visible | How common |
|---|---|---|
| Missed: vehicle seen from behind / above (driving towards or away from camera) | train_A (oblique interchange): 124 vehicles added on 6 frames, miss rate 0.60 | Very common on oblique views |
| Missed: small, far vehicles (10–25 px) | train_A, far parts of eval2 | Common |
| Missed: occluded or at frame edge | all clips | Occasional |
| False box: trees, kerbs, street furniture | train_D (urban), train_B | Common on busy scenes: ~1/3 of model boxes removed |
| False box: bridge structures, empty asphalt | eval2 | Occasional, but these boxes are large |
| False box: animal (cow) taken for a vehicle | eval2 | Rare |
| Partial box: one box covers half a vehicle, or a truck gets two boxes | eval (low-altitude nadir clip, vehicles ~250 px) | Very common on eval: 25 boxes redrawn, many removed/added pairs |
| Loose box (includes shadow, too wide) | all clips | Common, fixed by dragging corners (163 fixed boxes in total) |
| Two-wheelers (excluded by guide rule 1) | train_D | Removed from pre-labels after the first pass |

## Observations

- Pre-label quality depends mostly on **camera angle**: top-down simple highway (train_C) has 4% misses, the oblique interchange (train_A) 60%.
  COCO knows vehicles from the side, not from behind or above.
- **Partial boxes come from the labeler setup, not from the model:** SAHI tiles (frame width / 6) are smaller than close vehicles on the
  low-altitude eval clip, and the full-frame pass was off, so nothing merged the pieces.
- **Large false boxes count as "close" range:** distance is computed from box size, so spurious large boxes land in 0–200 m and inflate
  near-range false alarms.

## Pipeline log (decisions and problems, in order)

1. **First pre-label pass** (YOLO11x COCO + SAHI 640 px tiles, conf 0.25, car/truck/bus/motorcycle) missed most vehicles on train_A (1080p).
2. **Diagnosis on 4 train frames:** smaller tiles (frame width / 6) and conf 0.10 found more vehicles; motorcycles dropped (guide rule 1);
   full-frame pass turned off to remove big false boxes on trees. This became the final setting. Decided on train frames only.
3. **Review sample reduced** from 100 to 24 train frames because of time; eval kept at 60 uniform frames.
4. **CVAT issues:** "label not registered" (label missing in project), "Can't find image info" (images uploaded via a Drive folder download;
   fixed by uploading one flat zip built in Colab), "obj.data not found" (zip re-packed by the browser; fixed by downloading directly from Colab).
   Export came as "CVAT for images" XML; the pipeline parses it directly.
5. **Rule 4 check:** the corrected eval had 1 box in 200–400 m (nadir view from ~25–30 m). Added eval2
   (pexels.com/video/35695627, oblique highway): 121 boxes in 200–400 m. Same labeler setting, 30 uniform frames.
6. **Distance model fix before metrics:** HFOV 80° is applied to the long side of the frame (the eval clip is portrait).
7. **Metrics** computed once with all settings fixed. Main table in README; per-clip and IoU 0.3 runs in `results/`.
8. **Post-hoc run** (labeler with the full-frame pass on) was started but stopped: the Colab GPU quota ran out (5 of 60 frames done). Not reported.
9. **Videos** rendered from saved predictions on sampled eval frames (no GPU left to re-run the model on all frames).
