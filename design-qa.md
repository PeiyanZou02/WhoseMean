# Design QA

- Reference: `C:\Users\Peiyan\AppData\Local\Temp\codex-clipboard-f817daf1-dfd6-4f8a-a229-01e8348f3a68.png`
- URL: `http://127.0.0.1:8765/`
- Intended desktop viewport: 1416 × 664

## Visual targets

- Pure black field with X/Y/Z axes labelled warmth, luminance, and edge density.
- All 1,000 camera-facing source thumbnails visible in the spatial model and enlarged on hover.
- No line, ellipse, or path connecting thumbnails to the mean output.
- English-only controls and labels.
- Independent zoom for the entire XYZ space and the output image.
- Fixed right panel with live progress, mean outputs by epoch, and five-column generated results.

## Verified behavior

- Root HTML responds 200; it contains `START / RETRAIN`, contains no `LoRA`, and contains no prior `drawNetwork` function.
- `/api/works` returns exactly 1,000 works: 692 Prints, 176 Drawings, 69 Paintings, and 63 Photographs.
- The source thumbnail atlas is 1,280 × 3,200 and contains all 1,000 works in interface order.
- Embedded JavaScript and all relevant Python files pass syntax checks.
- The live training API completed one full 512 × 512 CUDA epoch over all 1,000 works in 34.63 seconds with train L1 0.08415.
- Epoch 0 and epoch 1 mean output frames were saved.
- Per-work generation completed for all 1,000 works and produced a 2,048 × 2,048 atlas.

## Remaining visual inspection limit

The Codex in-app browser controller could not be used because its runtime requested a missing cached `browser-service.mjs` version. HTTP behavior and generated images were inspected directly, but automated pointer interaction playback, console inspection, and screenshot comparison remain unavailable until that controller is repaired.
