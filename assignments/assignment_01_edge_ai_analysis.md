# Assignment 1 — Edge AI system analysis

**Chapter:** 1 — Edge AI system design
**Type:** Written analysis (no code required)
**Estimated effort:** 3-4 hours
**Submit as:** `experiments/reports/assignment_01_<your_name>.md` (or `.pdf`)

---

## Learning outcomes assessed

By submitting this assignment you demonstrate that you can:

1. Frame a real-world AI application as an Edge AI **system**, not a model choice.
2. Specify input, output, metric, latency budget, hardware, and failure modes for that application.
3. Justify why the application benefits from edge inference rather than cloud inference.

---

## Task

Pick **one** real-world Edge AI application. You may choose from the list in `SYLLABUS.md` (camera AI, sensor anomaly detection, industrial quality inspection, TinyML, Physical AI simulation) or propose your own. The application should:

- Run partly or fully on a device, not in the cloud.
- Have at least one constraint among: low latency, low bandwidth, privacy, offline operation, or physical control.
- Be specific enough that you can pick a sensor, a model class, and a deployment device.

Then write a system-design note using the template below. The note should be **1-2 pages** when formatted (about 800-1500 words). Concise and decisive beats long and hedged.

---

## Required template

Copy and fill in the following template. Do not skip sections; if a section does not apply, write "N/A" with a one-line reason.

```markdown
# System design: <application name>

## 1. Problem statement
One paragraph: what is being detected / classified / decided, by whom, in what
setting, with what business or operational consequence.

## 2. Why edge (not cloud)
One paragraph: which of latency / bandwidth / privacy / offline / physical
control is the dominant constraint, and why cloud-only is unacceptable.

## 3. Input
- Source: <camera / sensor / video stream / audio / file>
- Specs: resolution / rate / format / lighting / connection
- Rate: <Hz or FPS>
- Buffering: <every frame / every N / latest only>

## 4. Output
- Type: <label / box / mask / command / alert>
- Format: <JSON schema / MQTT topic / actuator command>
- Consumer: <UI / dashboard / actuator / log>

## 5. Metric
- Quality metric + target (be specific): <e.g. recall on defective ≥ 0.95 at
  precision ≥ 0.90>
- Performance metric + target: <e.g. P95 end-to-end ≤ 50 ms, FPS ≥ 20>
- Cost trade-off: <false positive cost vs false negative cost — explain which
  is worse and why>

## 6. Latency budget (P95, end-to-end)
| Stage | Budget (ms) |
|---|---|
| Capture | |
| Preprocess | |
| Model inference | |
| Postprocess | |
| Output / action | |
| Slack | |
| **Total** | |

State the total in one sentence: "End-to-end P95 budget is X ms, set by Y."

## 7. Hardware + runtime
- Device: <model / SoC, with link>
- Memory available: <MB after OS>
- Power envelope: <W>
- Runtime: <ONNX Runtime CPU / TensorRT / TFLite / ...>
- OS: <distro>
- Cost per unit (rough): <USD>

## 8. Failure modes + mitigations
At least 5 rows. Be concrete.

| Failure | What you detect | What the system does |
|---|---|---|
| Sensor disconnect | | |
| Low confidence | | |
| Data drift | | |
| Latency spike | | |
| Network outage | | |
| (others) | | |

## 9. Out of scope
List 3-5 things the system will explicitly NOT do.

## 10. Open questions
List 3-5 things you do not yet know that would change the design if you knew them.
```

---

## Grading rubric (100 points)

| Criterion | Points |
|---|---|
| Problem statement is concrete (specific user, setting, consequence) | 10 |
| "Why edge" cites a real constraint, not just "to be modern" | 10 |
| Input section names rate, format, and a realistic environment | 10 |
| Output is a real deliverable (not "the model's softmax"); consumer is named | 10 |
| Metric includes both quality AND performance AND a cost trade-off | 15 |
| Latency budget allocates across all stages and totals to a realistic number | 15 |
| Hardware + runtime match (no mismatched pairs like "RPi + large PyTorch model") | 10 |
| ≥5 failure modes with concrete detection and response | 10 |
| Out-of-scope and open-questions sections both filled with substantive entries | 5 |
| Note is concise (1-2 pages), well-formatted, decisive | 5 |
| **Total** | **100** |

---

## Common mistakes that lose points

- Starting the note with "I will use YOLOv8" or any other model name. The model is chosen *after* the spec, not before.
- Reporting a target accuracy without a target latency. Edge AI metrics are always plural.
- Latency budget that does not include capture or output stages — only "model inference."
- Failure modes that are only "the model is wrong." Sensor, network, thermal, and version failures all need entries.
- "Why edge" answered with "because it's faster" without naming the constraint (latency, bandwidth, privacy, offline, physical control).
- Hardware that cannot run the chosen model size — e.g., 1 GB model on a 512 MB RAM RPi.

---

## Suggested applications (if you need one)

You may pick any of the following or propose your own:

1. Real-time helmet detection on a construction site CCTV (worked partially in Ch 1, §9 — pick a different angle, e.g. PPE detection at a different site).
2. Predictive maintenance from a vibration sensor on an industrial motor.
3. Keyword spotting on a battery-powered wake-word device.
4. Product defect detection on a moving conveyor line.
5. People-counting at a retail store entrance.
6. Fall detection from an IMU on a wearable.
7. Pose-aware safety alert for a robotic arm cell.
8. ANPR (license plate recognition) at a parking lot gate.
9. Bird-call classification on a solar-powered field device.
10. Drone obstacle avoidance with a single forward camera.

---

## Submission checklist

Before submitting, verify:

- [ ] All 10 template sections are present and filled in (no missing headings).
- [ ] The latency budget table totals to a realistic number for your application.
- [ ] You have named **at least one** specific hardware device with a link or product code.
- [ ] You have named **at least one** specific runtime.
- [ ] You have listed **at least 5** failure modes.
- [ ] The note is 1-2 formatted pages, not 5-10.
- [ ] You did NOT pick the model first.
