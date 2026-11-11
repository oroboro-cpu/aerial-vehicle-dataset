# Labeling guide: class `vehicle`

Written before the correction pass and applied to all reviewed frames (train_small, eval, eval2).
Tool: CVAT (app.cvat.ai), one project with a single label `vehicle`.

## What to label

1. **Vehicle = 4+ wheels:** cars, vans, pickups, trucks, buses.
   Motorcycles, bicycles and scooters are **not** labeled. The distance estimate assumes a 4.5 m car,
   so a ~2 m motorcycle would be placed about 2x too far and land in the wrong range band.
2. **Parked vehicles** are labeled the same as moving ones.
3. **Occlusion:** label a vehicle if more than ~50% of it is visible. The box covers the visible part only.
   If it is about half visible or you are unsure, label it, and do it the same way on every frame.
4. **Frame edges:** vehicles cut by the frame border are labeled if rule 3 holds.
   (Their distance is overestimated because the box is shorter than the car.)

## How to draw

5. **Box:** tight around the body; the shadow is not included. A truck with trailer is one box.
6. **Minimum size (long side):** ≥ 10 px on 1080p, ≥ 20 px on 4K (≈ 500 m with our distance model).
   Smaller objects are beyond the 0–400 m bands and are not labeled. CVAT shows the box size while drawing;
   use the larger of the two numbers.
7. **Uncertain objects:** zoom in. If a person cannot tell at 2x zoom that it is a vehicle, do not label it.
   Motion blur is not a reason to skip a vehicle that is clearly there.

## Review order on each frame

8. 1) Delete false boxes (trees, roofs, road markings, structures, two-wheelers, people, animals).
   2) Fix loose or partial boxes (drag corners; ignore 1–2 px offsets).
   3) Scan far road areas left to right at high zoom and add missed vehicles.

## Sampling rules (set before review)

- Train: 6 frames per clip = 3 most suspicious (frame-to-frame jump in box count + low-confidence boxes) + 3 random.
- Eval and eval2: uniform frames only (every N-th frame), never "suspicious" frames, so that metrics are not biased.
- Eval frames get the most careful review: they are the ground truth.
