from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.evals.schema import EvalSample

_ALLOWED_QUERY_TYPES = {
    "single_hop_specific",
    "single_hop_abstract",
    "multi_hop_specific",
    "multi_hop_abstract",
}

_ABSTRACT_CUES = ("why", "how", "explain", "compare", "difference", "summar", "overview")


@dataclass
class QueryTypeIssue:
    severity: str
    code: str
    message: str


@dataclass
class QueryTypeValidationResult:
    sample_id: str
    query_type: str
    valid: bool
    issues: list[QueryTypeIssue]


def _collect_reference_file_ids(sample: EvalSample) -> list[str]:
    raw = sample.metadata.get("reference_file_ids") or []
    if isinstance(raw, str):
        raw = [raw]
    values = [str(item).strip() for item in raw if str(item).strip()]
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _contains_abstract_cue(text: str) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in _ABSTRACT_CUES)


class QueryTypeValidator:
    def __init__(self, mode: str = "warn", min_hop_evidence: int = 2) -> None:
        self.mode = mode
        self.min_hop_evidence = max(int(min_hop_evidence), 2)
        if self.mode not in {"off", "warn", "strict"}:
            raise ValueError(f"Unsupported validator mode: {mode}")

    def validate_sample(self, sample: EvalSample) -> QueryTypeValidationResult:
        issues: list[QueryTypeIssue] = []
        query_type = str(sample.query_type or "unknown").strip()

        if query_type not in _ALLOWED_QUERY_TYPES:
            issues.append(QueryTypeIssue("error", "unknown_query_type", f"query_type is not supported: {query_type}"))

        reasoning_hops = list(sample.reasoning_hops or [])
        hop_count = len(reasoning_hops)
        doc_evidence_count = len(set(sample.reference_doc_ids or []))
        file_evidence_count = len(set(_collect_reference_file_ids(sample)))
        effective_hop_evidence = max(hop_count, doc_evidence_count, file_evidence_count)

        is_multi = query_type.startswith("multi_hop_")
        is_single = query_type.startswith("single_hop_")

        if is_multi and effective_hop_evidence < self.min_hop_evidence:
            issues.append(
                QueryTypeIssue(
                    "error",
                    "insufficient_multihop_evidence",
                    f"multi-hop sample requires >= {self.min_hop_evidence} evidence units, got hops={hop_count}, docs={doc_evidence_count}, files={file_evidence_count}",
                )
            )

        if is_single and effective_hop_evidence >= self.min_hop_evidence:
            issues.append(
                QueryTypeIssue(
                    "error",
                    "singlehop_conflict_evidence",
                    f"single-hop sample has multi-hop-like evidence hops={hop_count}, docs={doc_evidence_count}, files={file_evidence_count}",
                )
            )

        user_text = str(sample.user_input or "")
        contains_abstract_cue = _contains_abstract_cue(user_text)
        is_abstract = query_type.endswith("_abstract")
        is_specific = query_type.endswith("_specific")

        if is_abstract and not contains_abstract_cue:
            issues.append(
                QueryTypeIssue("warn", "abstract_without_cue", "abstract query_type does not contain obvious abstract wording cues")
            )

        if is_specific and contains_abstract_cue:
            issues.append(
                QueryTypeIssue("warn", "specific_with_abstract_cue", "specific query_type appears to include abstract wording cues")
            )

        has_error = any(issue.severity == "error" for issue in issues)
        return QueryTypeValidationResult(
            sample_id=sample.sample_id,
            query_type=query_type,
            valid=not has_error,
            issues=issues,
        )

    def validate_samples(self, samples: list[EvalSample]) -> tuple[list[EvalSample], dict[str, Any]]:
        if self.mode == "off":
            return samples, {
                "mode": self.mode,
                "input_count": len(samples),
                "output_count": len(samples),
                "dropped_count": 0,
                "error_count": 0,
                "warning_count": 0,
                "issue_code_counts": {},
            }

        kept: list[EvalSample] = []
        dropped_count = 0
        issue_code_counter: Counter[str] = Counter()
        warning_count = 0
        error_count = 0

        for sample in samples:
            result = self.validate_sample(sample)
            validation_payload = {
                "mode": self.mode,
                "valid": result.valid,
                "issues": [
                    {
                        "severity": issue.severity,
                        "code": issue.code,
                        "message": issue.message,
                    }
                    for issue in result.issues
                ],
            }
            sample.metadata = dict(sample.metadata or {})
            sample.metadata["validation"] = validation_payload

            for issue in result.issues:
                issue_code_counter[issue.code] += 1
                if issue.severity == "error":
                    error_count += 1
                elif issue.severity == "warn":
                    warning_count += 1

            if self.mode == "strict" and not result.valid:
                dropped_count += 1
                continue
            kept.append(sample)

        summary = {
            "mode": self.mode,
            "input_count": len(samples),
            "output_count": len(kept),
            "dropped_count": dropped_count,
            "error_count": error_count,
            "warning_count": warning_count,
            "issue_code_counts": dict(sorted(issue_code_counter.items())),
        }
        return kept, summary
