# Aerial vehicle detection: dataset pipeline and range-split evaluation

Test task for thefourthlaw.ai. I focused on the data side: how frames are sampled, how model pre-labels are checked and corrected, which quality criteria apply, and how metrics are split into 0–200 m and 200–400 m. I did **not** train the on-board detector (< 500k parameters); see [Training](#training).

**Repo map:** `pipeline.ipynb` (all code, Google Colab) · `docs/labeling_guide.md` (labeling rules) · `docs/qa_notes.md` (error types seen in review) · `results/` (edit counts, distances, metrics, side runs) · `labels.zip` (YOLO format: `labels_fixed` = corrected labels of the 114 reviewed frames; `labels_raw` = pre-labels, final setting on the reviewed frames and first-pass setting on the other frames; `labels_raw_v1` = first-pass pre-labels) · `assets/` (GT vs prediction frames, videos, CVAT screenshots).

**Notebook guide:** `pipeline.ipynb` keeps the full working history, including superseded attempts. Final pipeline cells: 1 setup · 2 frames · 10 final pre-labels, review sample, CVAT packages · 21 eval2 · 24 helpers · 25 CVAT export → labels + edit counts · 26 distances · 27 metrics · 30 frames and videos. Superseded: 3–9 (first pass, diagnostics), 12–17 (earlier helpers, debugging), 28 (post-hoc run, stopped); the remaining cells are checks and re-runs after Colab restarts. CVAT correction is a manual step between cells 10 and 25.

| Rule | Status |
|---|---|
| 1. Eval out of every decision | All settings fixed on train frames before any eval metric; nothing changed after (see Eval setup) |
| 2. Labels from a model, then corrected | YOLO11x pre-labels corrected in CVAT; 329 added, 274 removed, 204 fixed/redrawn |
| 3. No aerial datasets | COCO weights only (no VisDrone/DOTA-trained weights) |
| 4. Both bands have data | Eval had 1 box in 200–400 m → added clip **eval2** (122 boxes in 200–400 m) |
| 5. Detector < 500k params | **Not met:** no detector trained; plan in Training |
| 6. README ≤ 4 pages | Logs and side runs are in `results/` and `docs/` |

## Data

| Clip | Role | Source | Resolution | Length | Sampled at | Frames extracted | Frames reviewed |
|---|---|---|---|---|---|---|---|
| train_A | train | pexels.com/video/8968356 | 1920×1080 | 19 s | 2 fps | 39 | 6 |
| train_B | train | pexels.com/video/5382494 | 3840×2160 | 31 s | 2 fps | 63 | 6 |
| train_C | train | pexels.com/video/8457857 | 3840×2160 | 17 s | 2 fps | 34 | 6 |
| train_D | train | pexels.com/video/3405804 | 3840×2160 | 24 s | 2 fps | 49 | 6 |
| eval | eval | pexels.com/video/32179597 | 2160×3840 (portrait) | 31 s | 5 fps | 153 | 60 |
| **eval2** | **eval, added (rule 4)** | [pexels.com/video/35695627](https://www.pexels.com/video/aerial-view-of-highway-in-vietnam-countryside-35695627/) | 3840×2160 | 44 s | 5 fps | 219 | 30 |

- Train clips are sampled at 2 fps (neighbouring frames at 30 fps are near-duplicates); eval clips at 5 fps for a denser time grid.
- **Why eval2.** The given eval clip is shot almost straight down from ~25–30 m. After correction it has 164 ground-truth boxes in 0–200 m and **1** in 200–400 m (median distance 27 m). eval2 is an oblique highway view looking towards the horizon (median 167 m, up to ~640 m). I made this decision from the ground-truth distance distribution, before computing any metric.

## Labels

### Pre-labeling
- **Model:** YOLO11x with COCO weights + SAHI tiled inference. COCO `car`, `truck`, `bus` → one class `vehicle`.
- **First pass** (640 px tiles, conf 0.25, motorcycles included) missed most vehicles on train_A (1080p): cars there are 10–25 px long, and COCO has few vehicles seen from above or behind.
- **Final setting, chosen on 4 train frames:** tile ≈ frame width / 6 (320 px for train_A, 640 px for the landscape 4K clips, 352 px for the portrait eval), 20% overlap, conf 0.10, full-frame pass off, motorcycles dropped. The first-pass labels are kept in `labels.zip` (`labels_raw_v1`) for comparison.

### Review sample
Correcting every frame did not fit the time budget, so I corrected a sample and measured the pre-labeler's error rate on it:
- **Train:** 6 frames per clip: the 3 most suspicious (largest frame-to-frame jump in box count + most low-confidence boxes) and 3 random. 24 frames.
- **Eval:** uniform frames only (60 eval + 30 eval2). Picking suspicious frames there would bias the metrics downwards.

### Quality criteria and process
I corrected the boxes in CVAT (app.cvat.ai, screenshot: `assets/cvat_screenshot.png`) following `docs/labeling_guide.md`, which I wrote **before** the review. The key rules:
- vehicle = 4+ wheels; two-wheelers are excluded because the distance model assumes a 4.5 m car;
- parked vehicles are labeled;
- a vehicle is labeled if more than 50% is visible;
- boxes are tight, without the shadow;
- minimum size on the long side is 10 px (1080p) / 20 px (4K), ≈ 500 m, just beyond the 400 m band edge.

Review order per frame: delete false boxes → fix loose boxes → scan the far road areas for missed vehicles.

### Edit counts
Each raw box is matched to the corrected box with the highest IoU:
- IoU ≥ 0.9: unchanged;
- 0.3–0.9: fixed;
- 0.1–0.3: *redrawn* (deleted and drawn again; usually a model box covering half a vehicle);
- no match: removed (raw box) / added (corrected box).

| Clip | Model boxes | Final boxes | Unchanged | Fixed | Redrawn | Removed | Added | Model precision | Model miss rate |
|---|---|---|---|---|---|---|---|---|---|
| train_A | 95 | 207 | 64 | 18 | 1 | 12 | 124 | 0.87 | 0.60 |
| train_B | 94 | 68 | 54 | 5 | 0 | 35 | 9 | 0.63 | 0.13 |
| train_C | 245 | 217 | 187 | 19 | 3 | 36 | 8 | 0.85 | 0.04 |
| train_D | 155 | 128 | 78 | 21 | 2 | 54 | 27 | 0.65 | 0.21 |
| eval | 157 | 165 | 12 | 63 | 25 | 57 | 65 | 0.64 | 0.39 |
| eval2 | 331 | 347 | 204 | 37 | 10 | 80 | 96 | 0.76 | 0.28 |
| **Total (114 frames)** | **1077** | **1132** | **599** | **163** | **41** | **274** | **329** | **0.75** | **0.29** |

### What the review showed
- **The scene and camera angle decide quality.** On the top-down simple highway (train_C) the pre-labeler missed 4% of vehicles. On the oblique interchange (train_A) it missed **60%**: COCO recognises vehicles seen from the side, but not vehicles seen from behind or above, i.e. traffic driving towards or away from the camera.
- **False boxes**, about a third of the model boxes on train_B/train_D: trees, kerbs and street furniture, bridge structures, empty asphalt, and once an animal (eval2).
- **Split vehicles.** On the low-altitude eval clip, vehicles (~250 px) are larger than the SAHI tile, and with the full-frame pass off the model often boxed **half a vehicle** or gave two boxes per truck. This explains most "redrawn" boxes on eval. It is a labeler configuration error (tile size vs object size), and I found it only while reviewing eval.
- **Extrapolation.** In the uncorrected train frames I expect roughly 25% false boxes and 30% missed vehicles, with oblique scenes much worse. These labels need correction before they can be used for training.

## Training

I did not train the on-board detector. This was a scope decision: I spent the ~8 hours on the data pipeline and label QA. With 24 corrected train frames, a trained model would give numbers that are hard to interpret.

How I would do it:
- Start from the YOLO11n config (~2.6M parameters) and reduce the width and depth multipliers until `model.info()` reports < 500k; report the final count and what was changed.
- Train at 1280 px input or on tiles. At 400 m a car is 26 px in a 4K frame, and only ~4 px after downscaling to 640.
- Hold out one train clip as validation, and choose epochs, checkpoint and confidence threshold on it only (rule 1).

**Weights:** there are no on-board detector weights. The labeler is the public `yolo11x.pt` from Ultralytics.

## Eval setup

- **What is measured:** the pre-labeler (YOLO11x + SAHI, final setting) against the corrected eval and eval2 frames: 90 frames, 512 ground-truth boxes. It is a reference point for a future small detector trained on these labels, not the on-board detector.
- **Distance:** pinhole model, D = f_px · 4.5 m / box long side (px), with f_px = (long frame side / 2) / tan(HFOV / 2). In 4K a car is 51 px long at 200 m and 26 px at 400 m.
- **Assumptions:**
  - car length 4.5 m (trucks and buses come out closer than they are);
  - HFOV 80° across the long side of the frame: a typical consumer drone, since Pexels gives no camera data. At 70° all distances grow by ~20%;
  - box long side ≈ vehicle length. This is wrong for vehicles foreshortened along the view direction or cut by the frame border: both come out too far;
  - slant range, not ground distance.
- **Matching:** IoU ≥ 0.5, greedy. TP and FN take the band of the ground-truth box, FP takes the band of the predicted box. Boxes beyond 400 m (9 GT) are not in the table.
- **False alarms per minute:** FP per frame × 5 fps × 60, i.e. alarms per minute for a detector running at 5 fps. The task formula FP × 60 / N_frames matches this only at 1 fps, so both are reported.
- **Rule 1:** confidence threshold, tiles, IoU, distance model and frame sample were all fixed before any eval metric was computed. Nothing was changed afterwards.

## Results

| Metric | 0–200 m | 200–400 m |
|---|---|---|
| Ground-truth boxes | 381 | 122 |
| Detection rate TP / (TP + FN) | **0.55** | **0.67** |
| Precision TP / (TP + FP) | **0.59** | **0.75** |
| False alarms / min (5 fps) | 486.7 | 93.3 |
| False alarms / min (task formula) | 97.3 | 18.7 |
| Time to first detection | not measured | not measured |

Per clip (`results/metrics_eval.csv`, `results/metrics_eval2.csv`):

| Clip | 0–200 m, DR / P | 200–400 m, DR / P |
|---|---|---|
| eval (nadir, ~27 m) | 0.37 / 0.39 | 1 box only |
| eval2 (oblique) | 0.69 / 0.74 | 0.67 / 0.75 |

**Reading the numbers:**
- **On the oblique clip the labeler is even across both bands.** The low 0–200 m score comes from the nadir clip, where tiles split large vehicles into partial boxes, and at IoU 0.5 each partial box counts as both an FP and an FN. So "close range is worse" is an artefact of the labeler setup, not of range.
- **False alarms are inflated in the near band.** An FP is assigned to a band by its own size, so large spurious boxes (on bridge structures, empty asphalt, an animal) land in 0–200 m. This is why near-range false alarms are high even on eval2 (example: `assets/eval2_false_boxes.png`).

These are the results of the setting fixed before evaluation, reported as they came out.

**Caveats:**
1. The ground truth was made by correcting this model's own output, so the numbers may be optimistic (anchoring).
2. IoU 0.5 is strict for 15–30 px boxes. A side run at IoU 0.3 is in `results/metrics_iou03.csv` and is not used above.

**Assets:**
- Frames (green = ground truth, red = model): `assets/eval_*_gt_vs_pred.jpg`, `assets/eval2_*_gt_vs_pred.jpg`.
- Videos: `assets/eval_predictions.mp4`, `assets/eval2_predictions.mp4`. They show the labeler's raw predictions on the sampled eval frames (the same frames used for metrics), coloured by distance band: green 0–200 m, orange 200–400 m, grey > 400 m. This is not an on-board detector. The videos were rendered from saved predictions because the Colab GPU quota ran out.

## What next

- **Fix the labeler tiling:** turn on the full-frame pass, or choose the tile size from the expected object size instead of the frame width. I found this problem while reviewing eval, so re-measuring with the fix would use eval-informed settings; that is why it is listed here and not in Results. A post-hoc run was started but stopped when the GPU quota ran out.
- **Better pre-labeler for top-down and rear views:** an open-vocabulary detector (OWLv2, YOLO-World), or a model fine-tuned on the corrected frames, then re-label (model in the loop).
- **Time to first detection:** this needs track IDs on consecutive frames. Label a 5–10 s eval segment in CVAT track mode and measure the time from first appearance to first detection.
- **More and better labels:**
  - correct more train frames;
  - have a second annotator re-label 10% of frames to measure agreement;
  - add more oblique, long-range footage (the 200–400 m band now comes almost entirely from one clip).
- **On-board detector:** as described in Training. The pixel budget drives the design: a < 500k model at 640 px input cannot see 4–9 px vehicles, so it needs high-resolution input or tiling.

**What I decided not to do:**
- train the < 500k detector;
- correct every frame;
- time to first detection;
- mAP@0.5.

**Use of AI:** I used an AI assistant (Claude) to write and debug the pipeline code and to draft this README. The labeling rules, all label corrections in CVAT, the review of the results and the decisions are mine.
