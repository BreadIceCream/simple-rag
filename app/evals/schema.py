from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _utcnow_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def detect_capabilities(payload: dict[str, Any]) -> list[str]:
    capabilities: list[str] = []
    if str(payload.get("reference_answer") or "").strip():
        capabilities.append("has_reference_answer")
    if payload.get("reference_contexts"):
        capabilities.append("has_reference_contexts")
    if payload.get("reference_doc_ids"):
        capabilities.append("has_reference_doc_ids")
    if payload.get("rubric"):
        capabilities.append("has_rubric")
    if payload.get("reference_tool_calls"):
        capabilities.append("has_reference_tool_calls")
    return capabilities


@dataclass
class EvalDatasetManifest:
    name: str
    version: str
    category: str
    source_type: str
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    review_required: bool = True
    review_instructions: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utcnow_iso)
    sample_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["capabilities"] = sorted(set(self.capabilities))
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalDatasetManifest":
        payload = dict(data)
        payload["capabilities"] = sorted(set(payload.get("capabilities") or []))
        payload["review_instructions"] = list(payload.get("review_instructions") or [])
        payload["metadata"] = dict(payload.get("metadata") or {})
        return cls(**payload)


@dataclass
class EvalSample:
    user_input: str
    sample_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    reference_answer: str | None = None
    reference_contexts: list[str] = field(default_factory=list)
    reference_doc_ids: list[str] = field(default_factory=list)
    scope_file_ids: list[str] = field(default_factory=list)
    difficulty_level: str = "unknown"
    scenario_type: str = "general"
    source_type: str = "manual"
    tags: list[str] = field(default_factory=list)
    review_status: str = "pending"
    review_notes: str = ""
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    rubric: dict[str, Any] | None = None
    reference_tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def normalize(self) -> "EvalSample":
        self.reference_contexts = [str(item).strip() for item in self.reference_contexts if str(item).strip()]
        self.reference_doc_ids = _ordered_unique([str(item).strip() for item in self.reference_doc_ids if str(item).strip()])
        self.scope_file_ids = _ordered_unique([str(item).strip() for item in self.scope_file_ids if str(item).strip()])
        self.tags = _ordered_unique([str(item).strip() for item in self.tags if str(item).strip()])
        merged_capabilities = list(self.capabilities) + detect_capabilities(asdict(self))
        self.capabilities = sorted(set(merged_capabilities))
        self.metadata = dict(self.metadata or {})
        self.reference_tool_calls = list(self.reference_tool_calls or [])
        return self

    def to_dict(self) -> dict[str, Any]:
        self.normalize()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalSample":
        payload = dict(data)
        if not payload.get("sample_id"):
            payload.pop("sample_id", None)
        payload["reference_contexts"] = list(payload.get("reference_contexts") or [])
        payload["reference_doc_ids"] = list(payload.get("reference_doc_ids") or [])
        payload["scope_file_ids"] = list(payload.get("scope_file_ids") or [])
        payload["tags"] = list(payload.get("tags") or [])
        payload["capabilities"] = list(payload.get("capabilities") or [])
        payload["metadata"] = dict(payload.get("metadata") or {})
        payload["reference_tool_calls"] = list(payload.get("reference_tool_calls") or [])
        sample = cls(**payload)
        return sample.normalize()


@dataclass
class EvalRunRecord:
    sample_id: str
    user_input: str
    dataset_name: str
    dataset_version: str
    category: str
    source_type: str
    scope_file_ids: list[str]
    actual_response: str = ""
    actual_contexts: list[str] = field(default_factory=list)
    actual_doc_ids: list[str] = field(default_factory=list)
    actual_file_ids: list[str] = field(default_factory=list)
    reference_answer: str | None = None
    reference_contexts: list[str] = field(default_factory=list)
    reference_doc_ids: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    difficulty_level: str = "unknown"
    scenario_type: str = "general"
    tags: list[str] = field(default_factory=list)
    review_status: str = "pending"
    latency_ms: float | None = None
    rewrite_count: int | None = None
    generate_count: int | None = None
    status: str = "success"
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["actual_contexts"] = [str(item) for item in self.actual_contexts]
        payload["actual_doc_ids"] = _ordered_unique(self.actual_doc_ids)
        payload["actual_file_ids"] = _ordered_unique(self.actual_file_ids)
        payload["reference_doc_ids"] = _ordered_unique(self.reference_doc_ids)
        payload["scope_file_ids"] = _ordered_unique(self.scope_file_ids)
        payload["capabilities"] = sorted(set(self.capabilities))
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalRunRecord":
        payload = dict(data)
        for key in (
            "scope_file_ids",
            "actual_contexts",
            "actual_doc_ids",
            "actual_file_ids",
            "reference_contexts",
            "reference_doc_ids",
            "capabilities",
            "tags",
        ):
            payload[key] = list(payload.get(key) or [])
        payload["metadata"] = dict(payload.get("metadata") or {})
        return cls(**payload)


def dataset_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "store" / "evals" / "datasets"


def dataset_dir(category: str, name: str, version: str, root: str | Path | None = None) -> Path:
    base = Path(root) if root else dataset_root()
    return base / category / name / version


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
