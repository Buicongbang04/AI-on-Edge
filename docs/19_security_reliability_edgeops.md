# Chapter 19 — Security, reliability, monitoring, and EdgeOps

> **Goal:** Teach what happens *after* a model "works in demo": versioning, logging, monitoring, fallback, secure update, rollback. By the end of this chapter you should be able to write a risk checklist for an edge AI system, design a fallback strategy, and define a device log schema.

Most teaching materials stop at "the model produces the right output". Production edge AI starts *after* that — when the model has to run for weeks on a remote device, get updated safely, fail predictably, and be auditable. This is **EdgeOps**.

---

## 1. What can go wrong

| Failure | Symptom | Detection |
|---|---|---|
| **Model error** | Wrong prediction at run time | Confidence threshold; cross-checks |
| **Sensor failure** | Black frame, frozen value, NaN | Sanity check on input statistics, watchdog |
| **Data drift** | Accuracy degrades over weeks | Monitor input distribution + sampled prediction agreement |
| **Latency spike** | Sporadic P99 blowups | Continuous P95/P99 monitoring |
| **Memory leak** | RSS grows over hours | Process-level health check + auto-restart |
| **Thermal throttling** | Latency suddenly worse on hot day | Read `tegrastats` / `vcgencmd`; throttle workload |
| **Power loss** | Brownout, mid-update corruption | Battery monitor; atomic update |
| **Wrong model deployed** | Bad build, wrong version | Model checksum + version logged on every prediction |
| **Network outage** | Cannot reach dashboard / MQTT | Local buffer + replay on reconnect |
| **Adversarial input** | Attacker-crafted noise / sticker | Out of scope here; flag as known limit |

Every one of these is the system's problem, not the model's problem. EdgeOps is the discipline of *catching them* before the user does.

---

## 2. Model versioning

Every deployed model artifact should travel with structured metadata:

| Field | Why |
|---|---|
| `model_version` | e.g. `mobilenetv3-small@v1.0.3` — appears on every log line |
| `runtime` | e.g. `onnxruntime==1.26.0+CPUExecutionProvider` — pinpoints what evaluator the prediction came from |
| `input_shape` | catches shape-mismatch deployments early |
| `preprocessing_config` | mean/std/resize — silent bugs hide here |
| `checksum_sha256` | guarantees the file on disk matches the one you intended to ship |
| `notes` | what changed since the last version |

The course's `src/edgeops/versioning.py` provides a `ModelVersion` dataclass and a SHA-256 helper. Use it for every model you deploy.

---

## 3. The device log schema

Every inference should be logged. The course's standard schema (matches Instruction.pdf §15):

```yaml
timestamp:            float        # Unix epoch
device_id:            str          # human-readable
model_version:        str          # from ModelVersion above
runtime:              str
input_source:         str          # 'camera' | 'sensor' | 'file' | ...
latency_ms:           float
confidence:           float        # 0..1
prediction:           str          # short text or class label
action_command:       str | null
temperature_celsius:  float | null
error_code:           str | null
fallback_triggered:   bool
```

JSONL (one JSON object per line) is the right format: append-only, line-streamable, parseable with `jq`. Rotate logs daily; ship a daily summary to a central dashboard.

The course's `src/edgeops/logging.py` provides a thread-safe `DeviceLogger`:

```python
from src.edgeops import DeviceLogger

with DeviceLogger("experiments/logs/device_42.jsonl",
                  device_id="device-42",
                  model_version="mobilenetv3-small@v1.0.3",
                  runtime="onnxruntime-CPU") as logger:
    logger.log(latency_ms=8.5, confidence=0.92, prediction="cat",
               input_source="camera")
```

---

## 4. Monitoring

What to watch in production:

- **Latency P95/P99 over time** — drift indicates thermal throttling or load contention.
- **Confidence distribution** — a sudden shift means the data drifted.
- **Prediction class histogram** — same warning.
- **Fallback trigger rate** — should be a few percent, not 30%.
- **Temperature, power, uptime** — system-level health.
- **Network reachability / queue depth** — for systems that publish events.

For small fleets, ship the JSONL daily to a central host and use `pandas` + a dashboard tool (Grafana, Streamlit). For larger fleets, push to a metrics backend (Prometheus / InfluxDB) or a managed log service.

---

## 5. Fallback strategies

The hierarchy of fallbacks, in priority order:

1. **Use prediction** if confidence is above threshold.
2. **Rule-based default** if confidence is medium (e.g. "do nothing", "go to safe state").
3. **Human review** if confidence is below the human-review threshold.
4. **Safe stop** for Physical AI (Ch 15 safety gate forces a no-op command).
5. **Hard stop / e-stop** when the system cannot decide at all.

The course provides `src/edgeops/fallback.py` with a small `confidence_fallback` helper that returns a structured decision the caller can act on.

For **Physical AI** specifically, the fallback hierarchy is enforced by the safety gate (Ch 15). For **camera AI** dashboards, fallback usually means "show 'low confidence' to the operator and queue for review".

---

## 6. Secure update

A remote device that runs your model needs a way to receive updates without bricking itself. The 2026 standard:

| Property | How |
|---|---|
| **Signed artifacts** | Sign the model file; verify signature before loading |
| **Atomic update** | Stage the new model; switch a symlink; rollback on failure |
| **Health check after switch** | Run a smoke test (e.g. inference on a held-out image); if it fails, revert |
| **Channel-based rollout** | Canary → small fleet → full fleet |
| **Versioned config** | Include the threshold, the labels, the preprocessing — they change too |
| **Time-stamped, reversible** | Keep the last 2-3 versions on disk for fast rollback |

Common tools (orientation only): **Mender**, **balena**, **Azure IoT Hub device twins**, **AWS IoT Greengrass**, **Foundries.io LmP**. They each implement the same pattern at different abstractions.

---

## 7. Rollback

When a new model version performs worse — confidence drops, fallback rate spikes, end-user complaints — the priority is **rollback to the last known good version** before debugging.

The minimum rollback plan:

1. Keep the previous N model artifacts on the device.
2. Track which one is "current" with a symlink or a config flag.
3. Switch the symlink → restart the inference process → verify with the health check.
4. Roll the change forward through the fleet, OR roll it back the same way you rolled it out.

If you cannot articulate the rollback steps in writing, you are not ready to ship the model.

---

## 8. The risk checklist (template)

Use this for every project's `deployment_notes/safety.md`:

```markdown
# Risk checklist — <project name>

## Failure modes
1. Sensor failure         — detection: ..., response: ...
2. Low confidence         — detection: confidence < 0.5, response: human_review
3. Data drift             — detection: prediction-class histogram diverges, response: alert + retrain
4. Latency spike          — detection: P95 > X ms over 1 min, response: throttle / skip frame
5. Memory leak            — detection: RSS doubles, response: restart process
6. Thermal throttling     — detection: temp > 80°C, response: drop input resolution
7. Wrong model deployed   — detection: checksum mismatch on load, response: refuse to start
8. Adversarial input      — out of scope; documented limitation

## Update plan
- Channel rollout: canary (1 device) → 10% → 100%, 24h between stages
- Health check: inference on `assets/canary.jpg` must produce class X with confidence ≥ Y
- Rollback: revert symlink to previous model version

## Operator override
- Hardware e-stop wired to safety relay (Physical AI only)
- Software override: set `human_override=true` in safety gate; device stops within one cycle
```

---

## 9. The course's worked example

`src/edgeops/` ships three helpers:

- `DeviceLogger`           — thread-safe JSONL writer with the standard schema.
- `ModelVersion`           — `from_file()` reads a model and computes its SHA-256.
- `confidence_fallback`    — picks `use_prediction` / `rule_based` / `human_review` based on thresholds.

Every project in the repo can drop in these helpers without writing custom infrastructure. The final project rubric (Ch 20) expects them to be used.

---

## 10. What you should be able to do after this chapter

- List the most common failure modes for your edge AI use case and the detection / response for each.
- Define a device log schema and use `DeviceLogger` to write structured logs.
- Compute and record a model artifact's SHA-256 checksum.
- Write a rollback plan that can be executed in <5 minutes.
- Pick a fallback strategy per use case (camera dashboard vs Physical AI).

---

## 11. Files produced by this chapter

- `docs/19_security_reliability_edgeops.md` — this file.
- `src/edgeops/__init__.py`, `logging.py`, `versioning.py`, `fallback.py` — the helpers.
- `deployment_notes/runtime.md`, `deployment_notes/safety.md` — templates required by Ch 20 final project rubric.
