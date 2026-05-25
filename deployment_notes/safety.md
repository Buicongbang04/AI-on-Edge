# Deployment notes: safety

**Template** — fill in one of these per project. The final-project rubric (Ch 20) requires this file to be present and complete.

---

## Project identification

| Field | Value |
|---|---|
| Project name | <e.g. helmet-detector-v2> |
| Owner | <name / team> |
| Last updated | <YYYY-MM-DD> |
| Physical AI? | <yes / no> — if yes, the controls below are mandatory; if no, parts are still advisable |

---

## Failure modes and responses

| # | Failure | Detection | Response |
|---|---|---|---|
| 1 | Sensor disconnect / black frame | Frame stats sanity check, watchdog | Use last good frame for K seconds; then alert "system offline" |
| 2 | Low model confidence | confidence < `min_confidence` (e.g. 0.5) | rule-based default; surface to operator |
| 3 | Very low model confidence | confidence < `require_human_below` (e.g. 0.3) | human review queue |
| 4 | Data drift | Prediction histogram diverges from baseline by > X% | Alert + retrain trigger |
| 5 | Latency spike | P95 over 30 s exceeds budget by 50% | Throttle / skip frame; alert |
| 6 | Thermal throttling | temp > 80 °C (Jetson) / 85 °C (RPi) | Drop input resolution; alert |
| 7 | Memory leak | RSS doubles over baseline | Restart process; alert |
| 8 | Network outage | Cannot reach broker for > 60 s | Buffer events locally; replay on reconnect |
| 9 | Wrong model deployed | Checksum mismatch at load | Refuse to start; revert to previous |
| 10 | Operator e-stop | Hardware switch pressed | All commands → (0, 0); persistent; require reset |

Add rows for project-specific failures.

---

## Physical AI safety layer (skip if not Physical AI)

Required checks in the safety gate (see `src/physical_ai/safety.py`):

| Check | Threshold | Action |
|---|---|---|
| Wheel / joint speed cap | <e.g. 0.7 m/s> | Clamp to limit |
| Hard-stop distance to obstacle | <e.g. 0.25 m> | Stop |
| Minimum perception confidence | <e.g. 0.5> | Stop; surface to operator |
| Human override | hardware switch | All commands → 0; persistent |
| Heartbeat timeout | <e.g. 500 ms> | Stop |
| Joint position limit | per joint | Clamp / abort |
| Time-to-collision | <e.g. 0.5 s> | Abort motion |

Reference: <link to ISO 13849 / ISO 10218 / industry safety standard you target>.

---

## Operator interaction

| Path | How |
|---|---|
| **Pause** | Software flag, single-line config change, takes effect within 1 cycle |
| **Stop** | Hardware e-stop wired to safety relay; removes power from motors |
| **Resume** | Manual reset required after any stop |
| **Override prediction** | Operator can mark an event as "false positive" or "missed defect"; goes into retrain queue |
| **Snooze alerts** | per-class snooze with audit log |

---

## Out of scope (security)

State explicitly what the system does NOT defend against. Examples:

- Adversarial image attacks (e.g. printed stickers crafted to fool the classifier).
- Network man-in-the-middle attacks on MQTT (unless you mandate TLS + auth).
- Physical tampering with the camera.
- Insider threats with shell access on the edge device.

If the project later needs these defenses, treat them as separate work items.

---

## Audit trail

- Every model load logs: `model_version`, `checksum_sha256`, `runtime`, `loaded_at`.
- Every inference logs: see `INFERENCE_LOG_SCHEMA` in `src/edgeops/logging.py`.
- Every safety verdict logs: `reason`, `blocked`, modified command.
- Every operator action logs: `who`, `what`, `when`.

Retention: at least 90 days local, indefinitely in a central log warehouse.
