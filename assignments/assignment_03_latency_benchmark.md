# Assignment 3 — Honest latency benchmark

**Chapter:** 4 — Benchmarking
**Type:** Code + short report
**Estimated effort:** 3-5 hours
**Submit as:** `experiments/reports/assignment_03_<your_name>.md` + the JSON results in `experiments/benchmark_results/`.

---

## Learning outcomes assessed

By submitting this assignment you demonstrate that you can:

1. Run the `src.benchmark` toolkit on a real model and produce a structured report.
2. Vary at least 2 axes (batch size, device, model variant, or input size) and compare results.
3. Distinguish model-only latency from end-to-end latency, and identify the bottleneck.
4. Read tail percentiles (P95 / P99) and explain what they imply for deployment.

---

## Task

### Required

Pick **one** image classifier from torchvision (or your own model — must run in PyTorch).

Run the standard course benchmark across **at least two** of the following axes (so at least 4 cells in your comparison table):

- **Device:** CPU vs CUDA (if you have a GPU).
- **Batch size:** 1, 4, 16.
- **Model variant:** e.g. MobileNetV3-Small vs MobileNetV3-Large, ResNet18 vs ResNet50.
- **Input resolution:** 160×160 vs 224×224 vs 320×320.

For each cell, save the `experiments/benchmark_results/<name>-<timestamp>.json` file produced by `python -m src.benchmark`, AND report the key metrics in the table below.

### Required table (in your report)

```markdown
| Run | Model | Device | Batch | Input | params (M) | model_only P50 (ms) | model_only P95 (ms) | end_to_end P95 (ms) | mem peak (MB) | FPS end_to_end |
|-----|-------|--------|-------|-------|------------|---------------------|---------------------|---------------------|---------------|----------------|
| 1   |       |        |       |       |            |                     |                     |                     |               |                |
| 2   |       |        |       |       |            |                     |                     |                     |               |                |
| 3   |       |        |       |       |            |                     |                     |                     |               |                |
| 4   |       |        |       |       |            |                     |                     |                     |               |                |
```

### Required short analysis (≤500 words)

Answer all of the following in your report:

1. **Which configuration achieved the best end-to-end FPS?** And which P95?
2. **Where is the bottleneck?** Compare `model_only P95` to `end_to_end P95`. If they are close, the model is the bottleneck; if `end_to_end` is much higher, capture/preprocess/postprocess is the bottleneck.
3. **What is the largest single optimization you could try next?** (Hint: Chapters 6-8 will show you several. Pick one and predict what it will buy.)
4. **Tail latency:** Is P95 ≤ 1.5 × P50, or larger? A larger ratio means more jitter — what would cause it on your machine?
5. **Memory:** Did peak memory grow with batch size? Could you fit batch=16 on your target edge device's RAM?
6. **Honest disclosure:** What did you NOT control for in this benchmark? (Background processes, thermal state, network activity, power source, etc.)

---

## Steps

1. Read `docs/04_benchmarking_and_profiling.md`.
2. Run the CLI to produce one baseline:

   ```bash
   python -m src.benchmark --model mobilenet_v3_small --device cpu --pipeline
   ```

3. Vary one axis at a time:

   ```bash
   python -m src.benchmark --model mobilenet_v3_small --device cpu --batch-size 4 --pipeline
   python -m src.benchmark --model mobilenet_v3_large --device cpu --pipeline
   python -m src.benchmark --model resnet18         --device cpu --pipeline
   ```

4. (If you have CUDA) repeat at least one row on `--device cuda`.

5. Inspect the JSONs in `experiments/benchmark_results/`. Pull the numbers into your report table.

6. Write the analysis section.

---

## Grading rubric (100 points)

| Criterion | Points |
|---|---|
| Ran the benchmark on ≥4 configurations across ≥2 axes | 20 |
| Saved JSON for every run, in `experiments/benchmark_results/` | 10 |
| Report table is filled in with the required columns | 15 |
| Reported model-only AND end-to-end latency (not just one) | 10 |
| Reported P50 AND P95 (not just mean) | 10 |
| Answered all 6 analysis questions | 20 |
| Identified the bottleneck correctly with evidence | 10 |
| Honest about what was NOT controlled | 5 |
| **Total** | **100** |

---

## Common mistakes that lose points

- Reporting only `mean`. Tail percentiles are required.
- Not running with warm-up. The first iteration is not steady-state.
- Mixing CPU and CUDA numbers in the same row without labels.
- Comparing batch=1 latency to batch=16 latency without dividing by batch — they are not the same number.
- Not running the `--pipeline` flag at least once. You need an end-to-end FPS row.
- Reporting peak memory without subtracting the baseline.

---

## Stretch goals (not graded, but interesting)

- Repeat one configuration twice and compute the std deviation across runs (not just within a run).
- Run a 5-minute sustained loop and plot latency over time. Did it drift?
- If you have a Jetson or Pi available, run the same model and compare. The latency gap will be more illuminating than any synthetic comparison.
