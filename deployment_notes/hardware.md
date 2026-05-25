# Deployment notes: hardware

**Template** — fill in one of these per project. The final-project rubric (Ch 20) requires this file alongside `runtime.md` and `safety.md`. This file captures *physical* deployment decisions (device choice, mounting, power, peripherals, environment). Runtime-level config goes in `runtime.md`; failure handling goes in `safety.md`.

---

## Identification

| Field | Value |
|---|---|
| Project name | <e.g. helmet-detector-v2> |
| Owner | <name / team> |
| Last updated | <YYYY-MM-DD> |
| Site / line | <e.g. Hanoi factory, line 3> |
| Number of devices in fleet | <e.g. 4 production + 1 canary> |

---

## Device selection

| Field | Value |
|---|---|
| Device family | <e.g. NVIDIA Jetson Orin Nano 8GB Developer Kit> |
| Reference notes | See `hardware_notes/nvidia_jetson.md` |
| Reason chosen | <e.g. Need TensorRT + sustained 15 FPS on YOLOv8n@640; RPi 5 too slow, Orin AGX over-budget> |
| Alternatives ruled out | <e.g. RPi 5 + Coral USB (operator support gaps); Intel NUC (no GPU, OpenVINO YOLO slower)> |
| Unit cost | <USD> |
| Lead time | <weeks> |
| Replacement SKU / supplier | <vendor + part number> |

Cross-reference the Hardware–Runtime–Use case matrix in Instruction §12 to justify the choice.

---

## Physical setup

| Field | Value |
|---|---|
| Enclosure | <e.g. IP54 metal box, fan-cooled> |
| Mounting | <e.g. DIN rail in panel; 2.5 m above conveyor> |
| Orientation | <e.g. camera tilted 15° down> |
| Cable routing | <e.g. shielded Cat6 + USB 3.0 active extension, max 5 m> |
| Vibration isolation | <none / rubber mounts / shock-absorbing bracket> |
| Access for maintenance | <e.g. front panel hinged; SD card reachable without disassembly> |

Attach a photo and a simple layout sketch in `figures/` if helpful.

---

## Power

| Field | Value |
|---|---|
| Input voltage | <e.g. 12 V DC from regulated PSU> |
| Power supply rating | <e.g. 65 W, 80+ Gold> |
| Device power mode | <e.g. Jetson `nvpmodel -m 0` (MAXN_SUPER), 25 W budget> |
| Measured power draw (idle / active / peak) | <e.g. 6 / 14 / 22 W> |
| UPS / battery backup | <yes / no — runtime if yes> |
| Brown-out behavior | <e.g. systemd auto-restart on power return; last config persisted to disk> |

---

## Thermal

| Field | Value |
|---|---|
| Cooling | <passive heatsink / active fan / forced air> |
| Ambient temperature range | <e.g. 5–35 °C> |
| Measured steady-state device temp | <e.g. 62 °C @ 25 °C ambient, sustained load> |
| Thermal throttle threshold | <e.g. Jetson throttles ≥ 85 °C → see safety.md row 6> |
| Dust / IP rating | <e.g. IP54 enclosure; clean filter monthly> |

---

## Storage

| Field | Value |
|---|---|
| Boot media | <e.g. NVMe SSD 256 GB; SD card is recovery only> |
| Filesystem layout | <e.g. `/` root, `/var/log` separate partition, `/opt/edge-ai` for models> |
| Free space at deploy | <GB> |
| Log retention policy | <e.g. JSONL inference logs rotated daily, 7 days local, then shipped to log warehouse> |
| Wear concerns | <e.g. SD card lifetime ~2 years at current write rate — plan refresh> |

---

## Network

| Field | Value |
|---|---|
| Link | <Ethernet / Wi-Fi / 4G LTE> |
| Bandwidth budget | <e.g. ≤ 50 kbps uplink for telemetry; alerts only, no frames> |
| MQTT broker / API endpoint | <host:port> |
| Offline tolerance | <e.g. queues events locally up to 24 h, replays on reconnect> |
| Time sync | <NTP server / cellular> — required for correct log timestamps |
| Remote access | <e.g. Tailscale SSH, no public IP> |

---

## Connected peripherals

### Cameras / image sensors

| Field | Value |
|---|---|
| Model | <e.g. IMX477, Hikvision DS-2CD2143G2-I> |
| Interface | <CSI / USB 3 / GigE> |
| Resolution / FPS | <e.g. 1920×1080 @ 30 FPS> |
| Lens / FoV | <e.g. 6 mm, 60° HFOV> |
| Lighting assumption | <e.g. >200 lux, no direct sunlight on lens> |
| Calibration file | <path or n/a> |

### Other sensors / actuators

| Device | Interface | Sample rate | Notes |
|---|---|---|---|
| <e.g. vibration sensor> | I²C @ 0x68 | 1 kHz | mounted on motor housing |
| <e.g. relay output> | GPIO 17 | event-driven | controls alert beacon |

---

## Environment

| Field | Value |
|---|---|
| Indoor / outdoor | <e.g. indoor, climate-controlled> |
| EM / RF interference | <e.g. nearby VFDs; shielded cabling required> |
| Hazards | <e.g. food-grade environment — wipe-down required; explosive atmosphere — not applicable> |
| Operating hours | <e.g. 24/7 / 2-shift / weekday business hours> |

---

## Bill of materials

| Item | Qty | Part # / source | Unit cost | Notes |
|---|---|---|---|---|
| Compute device | 1 | <e.g. Jetson Orin Nano 8GB Dev Kit, 945-13766-0005-000> | | |
| Camera | 1 | | | |
| Lens | 1 | | | |
| Power supply | 1 | | | |
| Enclosure | 1 | | | |
| Cabling | | | | |
| Mounting bracket | 1 | | | |
| Storage (NVMe / SD) | 1 | | | |
| **Total per device** | | | | |

---

## First-boot / provisioning

1. Flash OS image: `<JetPack version / RPi OS version / image hash>`.
2. Apply baseline config: `<ansible playbook / script path>` (sets hostname, timezone, SSH keys, NTP).
3. Install runtime dependencies: `<pip install -r requirements-device.txt>` or container pull.
4. Deploy model artifact (see `runtime.md` §"Deployment process").
5. Register device: POST to `<endpoint>` with `device_id`, `hardware_serial`, `model_version`.
6. Run on-device acceptance test: `<script>` — must pass before site handover.

---

## Site acceptance checklist

- [ ] Device powers on and reaches steady state within 90 s
- [ ] Camera feed visible at `<URL>` and not flipped/rotated
- [ ] Inference loop logging to `/var/log/edge-ai/` with correct `device_id`
- [ ] Steady-state device temperature < <X> °C after 30 min load
- [ ] Network reachability to broker / API verified
- [ ] Heartbeat visible on monitoring dashboard `<link>`
- [ ] Rollback plan tested (`runtime.md` §"Rollback plan")
- [ ] Safety gate tested per `safety.md` failure modes 1, 5, 6, 9, 10
- [ ] Operator trained on pause/stop/resume procedures
- [ ] Spare device and spare SD/NVMe stored on-site

---

## Maintenance schedule

| Interval | Task |
|---|---|
| Daily | Verify dashboard heartbeat; review alerts |
| Weekly | Inspect lens for dust/condensation; review fallback rate |
| Monthly | Clean enclosure filter; review log volume vs retention |
| Quarterly | Full health check script; rotate canary device |
| Yearly | Storage wear check; thermal paste / fan inspection on Jetson |
