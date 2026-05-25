# Chapter 18 — Edge LLM and multimodal AI on devices

> **Goal:** Understand the state of on-device large language and vision-language models in 2026, why they matter, what works, and what does not. This chapter is **advanced** — it is positioned at the end of the syllabus on purpose. Newcomers to Edge AI should not start here.

Running LLMs on the device unlocks privacy, offline use, low-latency assistants, and robotics interfaces. It also collides with the hardest edge constraint (memory). The chapter is *survey + decision framework*, not implementation-heavy. The course's notebook runs a small quantized model on the laptop and demonstrates the workflow; real edge LLM deployment depends on having matching hardware (Apple Silicon, Intel Core Ultra NPU, Qualcomm SoC, Jetson Orin AGX).

---

## 1. The 2026 landscape

| Model class | Where it runs on the edge | Typical edge constraint |
|---|---|---|
| 1-3 B parameter SLMs (Gemma 2B, Phi-3-mini, Qwen-2.5 1.5B) | Apple Silicon, Intel Core Ultra NPU, Jetson Orin Nano, beefy laptops | RAM (4-6 GB), latency (tokens/s) |
| 7-8 B parameter LLMs (Llama-3-8B, Mistral-7B, Qwen-2.5-7B) | Apple M-series, Intel Core Ultra NPU + offload, Jetson Orin NX/AGX, NUCs with discrete GPU | RAM (8-12 GB quantized) |
| 70 B+ LLMs | Not edge in 2026 | Beyond device RAM |
| Small VLMs (Phi-3-Vision, Florence, Moondream) | Laptops, M-series, Orin NX/AGX | RAM + image encoder cost |
| Robot VLA models (NVIDIA Isaac GR00T N1.6, OpenVLA) | Jetson Orin NX/AGX, server-class robotics | Specialized; emerging |

**The course's safe edge LLM target in 2026:** quantized 1-3 B parameter models on a laptop (Apple Silicon or Intel Core Ultra NPU), or on a Jetson Orin Nano. Anything bigger likely needs Orin NX/AGX or a NUC with discrete GPU.

---

## 2. Why on-device LLM

Three motivations:

1. **Privacy:** prompts and responses never leave the device. Critical for healthcare, legal, and personal-assistant use cases.
2. **Latency:** no network round-trip; predictable response time, especially for voice interfaces.
3. **Offline capability:** works on a robot in a remote field, on a plane, behind a firewall.

Three downsides:

1. **Quality gap:** an on-device 3B-class model is meaningfully worse than a cloud 70B+ model on hard tasks. Pick on-device only when "good enough" is good enough.
2. **Memory:** even quantized, 7-8B models need 6-10 GB of RAM. Many edge devices do not have it.
3. **Energy:** generating long responses on the device drains battery. Streaming and early-stopping matter.

---

## 3. Quantization for LLMs

LLM quantization is more aggressive than CV quantization (Ch 8):

| Scheme | Bits / weight | Bits / activation | Use case |
|---|---|---|---|
| FP16 / BF16 | 16 | 16 | Server / desktop GPU baseline |
| INT8 (W8A8) | 8 | 8 | Edge with INT8 NPU/TPU |
| INT4 (W4A16 / W4A8) | 4 | 16 / 8 | Mainstream local LLM deployment (GGUF, AWQ, GPTQ) |
| INT3 / INT2 | 3 / 2 | varies | Experimental; meaningful quality loss |

For a 7B model: FP16 = ~14 GB, INT8 = ~7 GB, INT4 = ~4 GB. The INT4 path is what makes 7B-class models fit on consumer laptops and Jetson Orin Nano.

Popular formats:

- **GGUF** (used by `llama.cpp`) — the de facto format for local LLM inference; quantized variants like `Q4_K_M`, `Q5_K_M`.
- **MLX** — Apple Silicon-native format.
- **AWQ / GPTQ** — quantization methods, often stored as Hugging Face checkpoints.

---

## 4. Runtimes for edge LLM

| Runtime | Where it runs | Notes |
|---|---|---|
| **llama.cpp** | Anywhere (CPU + Metal + CUDA + Vulkan + ROCm) | Most portable; reads GGUF |
| **Ollama** | Wraps llama.cpp with a server + CLI | Easiest local experience |
| **MLX** | Apple Silicon | Best Mac performance |
| **OpenVINO 2026** | Intel CPU + iGPU + NPU | Strong on Core Ultra NPU for attention layers |
| **TensorRT-LLM** | NVIDIA GPU / Jetson | Highest perf on NVIDIA; requires NVIDIA stack |
| **vLLM** | Server-class, often not edge | Mentioned for completeness |

For the course's notebook, **Ollama** or **llama.cpp** with a GGUF quantized model is the recommended path. They work on Linux / macOS / Windows.

---

## 5. Prompt and token latency

Two latency numbers matter:

- **Time-to-first-token (TTFT)** — wait before the first character appears. Dominated by prompt-encoding (prefill).
- **Tokens-per-second (TPS)** — sustained generation speed. Dominated by decode.

For a 3B model at INT4 on a 16 GB Mac M-series: TTFT ~0.2-0.5 s, TPS ~20-50.
For a 7B model at INT4 on Jetson Orin NX: TTFT ~1 s, TPS ~15-25.
For the same 7B model on Intel Core Ultra NPU with OpenVINO 2026: TPS ~25-40 (≈3.8× over GPU-only on the same box).

For interactive applications, target **TTFT < 1 s** and **TPS ≥ 10**. Below that, users notice the lag.

---

## 6. Memory footprint math

Quick formula for an autoregressive LLM at inference time:

```
RAM ≈ weights + KV cache + activations + framework overhead

weights        = params × bytes_per_weight
KV cache       = layers × seq_len × hidden_size × 2 × bytes_per_kv_element
activations    = a few × hidden_size × seq_len × bytes_per_act
overhead       = 200-500 MB for framework
```

For a 7B model, 32 layers, hidden=4096, 4-bit weights, 8-bit KV cache, 4K context:

- weights ≈ 7e9 × 0.5 B = 3.5 GB
- KV cache ≈ 32 × 4096 × 4096 × 2 × 1 B = 1 GB
- overhead ≈ 0.5 GB
- **Total ≈ 5 GB** — fits on an 8 GB device with breathing room.

If you double the context length, the KV cache doubles. Long-context LLMs are the most RAM-hungry part of the system, not the weights.

---

## 7. Vision-language models (VLMs)

VLMs add an image encoder in front of an LLM. The encoder is usually a CLIP-style ViT; the LLM is the same family as text-only models.

- **Florence-2 (Microsoft, 230M-770M)** — small, fast, good for OCR / detection prompts.
- **Phi-3-Vision** — text + image, ~4B params, reasonable on Jetson Orin Nano.
- **Moondream** — very small VLM (~2B); designed for edge.
- **NVIDIA Cosmos-Reason** — large VLM-for-physical-AI; not edge.

For an edge VLM application:

- Run the vision encoder *once* per frame.
- Cache the visual embedding.
- Run the language head with multiple prompts against the same cached embedding.

This is the pattern for "camera assistants" — "Describe what you see", "Is there a person in this frame?".

---

## 8. Edge LLM for robotics: VLA models

A vision-language-action (VLA) model is a VLM extended to output **actions** (or action distributions) rather than just text. NVIDIA's Isaac GR00T N1.6 is the flagship 2026 open VLA for humanoid robots.

For this course's scope, VLA is **orientation only** — running GR00T or OpenVLA on the edge needs server-class GPUs or Orin AGX. The relevant message: VLA is the future of the *decision* layer in Physical AI; the loop architecture (Ch 14) is unchanged.

---

## 9. The course notebook

`notebooks/chapter_11_edge_llm_intro.ipynb` walks through a **concept demo**:

- Shows the memory math for a chosen model.
- Demonstrates a Hugging Face `transformers` text-generation pipeline on a small model (TinyLlama 1.1B, Qwen-2.5-0.5B, or similar) — works on a laptop CPU/GPU.
- Compares TTFT and TPS at different quantization levels (when a quantized backend like `bitsandbytes` is installed).
- Discusses what would change on Apple Silicon, Intel NPU, or Jetson.

The notebook is set up to **skip gracefully** if you do not want to download multi-GB model weights.

---

## 10. When to use cloud LLM instead

For most edge AI applications, the right answer is *still* a cloud LLM. Use on-device only when:

- Privacy or compliance requires it.
- Network is unreliable or absent.
- Latency budget is sub-second and predictable.
- Cost at scale (millions of devices) favors local inference.
- The use case is *focused* and a 1-7B model can handle it.

For long-tail reasoning, coding, agentic workflows, or anything where a 70B+ model is materially better, prefer cloud.

---

## 11. What you should be able to do after this chapter

- Estimate whether a given LLM fits on a given device (weights + KV + overhead).
- Pick a quantization scheme and runtime for a target hardware.
- Distinguish TTFT from TPS and articulate which matters for your application.
- Decide between an edge LLM and a cloud LLM for a use case.
- Sketch a camera → vision model → local LLM decision pipeline.

---

## 12. Files produced by this chapter

- `docs/18_edge_llm_multimodal_ai.md` — this file.
- `notebooks/chapter_11_edge_llm_intro.ipynb` — concept walkthrough with a small HF model.
