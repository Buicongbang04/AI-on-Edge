"""Model versioning helpers (Chapter 19).

Every deployed model should travel with a structured `ModelVersion` record so
that logs and benchmarks can be matched back to a specific artifact.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


def compute_model_checksum(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Return the SHA-256 checksum of a model file."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ModelVersion:
    """Structured deployment metadata for a single model artifact.

    Required fields (per Instruction.pdf §15):
        model_version, runtime_version, input_shape, preprocessing_config, checksum
    """
    name: str                         # e.g. "mobilenet_v3_small"
    model_version: str                # e.g. "v1.0.3" or git SHA
    runtime: str                      # e.g. "onnxruntime==1.26.0+CPUExecutionProvider"
    input_shape: tuple[int, ...]      # e.g. (1, 3, 224, 224)
    preprocessing_config: dict[str, Any]   # mean/std/resize/crop
    checksum_sha256: str              # SHA-256 of the model artifact
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_file(
        cls,
        artifact_path: str | Path,
        *,
        name: str,
        model_version: str,
        runtime: str,
        input_shape: tuple[int, ...],
        preprocessing_config: dict[str, Any],
        notes: str = "",
    ) -> "ModelVersion":
        return cls(
            name=name,
            model_version=model_version,
            runtime=runtime,
            input_shape=tuple(input_shape),
            preprocessing_config=preprocessing_config,
            checksum_sha256=compute_model_checksum(artifact_path),
            notes=notes,
        )
