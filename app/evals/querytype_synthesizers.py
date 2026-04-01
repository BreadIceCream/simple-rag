from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from app.evals.build_synthetic_dataset import _generate_with_chunks, _to_rows

QUERY_TYPES = (
    "single_hop_specific",
    "single_hop_abstract",
    "multi_hop_specific",
    "multi_hop_abstract",
)

QUERY_DISTRIBUTION_PROFILES: dict[str, dict[str, float]] = {
    "balanced": {
        "single_hop_specific": 0.25,
        "single_hop_abstract": 0.25,
        "multi_hop_specific": 0.25,
        "multi_hop_abstract": 0.25,
    },
    "multihop_focus": {
        "single_hop_specific": 0.10,
        "single_hop_abstract": 0.10,
        "multi_hop_specific": 0.40,
        "multi_hop_abstract": 0.40,
    },
}


def normalize_query_distribution(raw_distribution: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for query_type in QUERY_TYPES:
        value = raw_distribution.get(query_type, 0.0)
        try:
            normalized[query_type] = max(float(value), 0.0)
        except (TypeError, ValueError):
            normalized[query_type] = 0.0
    total = sum(normalized.values())
    if total <= 0:
        return dict(QUERY_DISTRIBUTION_PROFILES["multihop_focus"])
    return {query_type: value / total for query_type, value in normalized.items()}


def resolve_query_distribution(
    *,
    profile: str,
    distribution_json: str | None = None,
    distribution_file_payload: dict[str, Any] | None = None,
) -> dict[str, float]:
    if distribution_json:
        import json

        return normalize_query_distribution(json.loads(distribution_json))
    if distribution_file_payload:
        return normalize_query_distribution(distribution_file_payload)
    if profile not in QUERY_DISTRIBUTION_PROFILES:
        raise ValueError(f"Unsupported query distribution profile: {profile}")
    return dict(QUERY_DISTRIBUTION_PROFILES[profile])


def allocate_query_type_counts(total: int, distribution: dict[str, float]) -> dict[str, int]:
    safe_total = max(int(total), 0)
    if safe_total == 0:
        return {query_type: 0 for query_type in QUERY_TYPES}

    normalized = normalize_query_distribution(distribution)
    allocations: dict[str, int] = {}
    remainders: list[tuple[str, float]] = []
    assigned = 0
    for query_type in QUERY_TYPES:
        exact = safe_total * normalized[query_type]
        base = int(exact)
        allocations[query_type] = base
        assigned += base
        remainders.append((query_type, exact - base))

    for query_type, _ in sorted(remainders, key=lambda item: item[1], reverse=True):
        if assigned >= safe_total:
            break
        allocations[query_type] += 1
        assigned += 1
    return allocations


def _load_synthesizer_class(class_name: str) -> type[Any] | None:
    module_candidates = [
        "ragas.testset.synthesizers",
        "ragas.testset.synthesizers.single_hop",
        "ragas.testset.synthesizers.multi_hop",
    ]
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        candidate = getattr(module, class_name, None)
        if candidate is not None:
            return candidate
    return None


def _build_distribution_entry(class_name: str) -> Any | None:
    synthesizer_cls = _load_synthesizer_class(class_name)
    if synthesizer_cls is None:
        return None
    try:
        return (synthesizer_cls(), 1.0)
    except Exception:
        try:
            return synthesizer_cls()
        except Exception:
            return None


def _invoke_with_distribution(
    generator: Any,
    *,
    chunks: list[Any],
    testset_size: int,
    run_config: Any,
    query_distribution: list[Any] | None,
) -> list[dict[str, Any]]:
    if query_distribution:
        attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
            ((), {"chunks": chunks, "testset_size": testset_size, "run_config": run_config, "query_distribution": query_distribution}),
            ((), {"chunks": chunks, "size": testset_size, "run_config": run_config, "query_distribution": query_distribution}),
            ((chunks, testset_size), {"run_config": run_config, "query_distribution": query_distribution}),
        ]
        for args, kwargs in attempts:
            try:
                testset = generator.generate_with_chunks(*args, **kwargs)
                return _to_rows(testset)
            except TypeError:
                continue
    testset = _generate_with_chunks(generator, chunks, testset_size, run_config)
    return _to_rows(testset)


def _normalize_rows(rows: list[dict[str, Any]], query_type: str, query_type_source: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        current = dict(row)
        current["query_type"] = query_type
        current.setdefault("query_type_source", query_type_source)
        normalized.append(current)
    return normalized


def _adapt_single_hop_abstract_rows(base_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adapted: list[dict[str, Any]] = []
    for row in base_rows:
        current = dict(row)
        question = str(current.get("user_input") or current.get("question") or current.get("query") or "").strip()
        if question:
            stem = question.rstrip(" ?")
            current["user_input"] = f"Explain the core idea behind: {stem}."
        current["query_type"] = "single_hop_abstract"
        current["query_type_source"] = "project_single_hop_abstract_adapter"
        adapted.append(current)
    return adapted


@dataclass
class QueryTypeSynthesisResult:
    rows: list[dict[str, Any]]
    requested_counts: dict[str, int]
    generated_counts: dict[str, int]
    source_map: dict[str, str]


class QueryTypeSynthesizerFacade:
    _CLASS_NAME_BY_TYPE = {
        "single_hop_specific": "SingleHopSpecificQuerySynthesizer",
        "multi_hop_specific": "MultiHopSpecificQuerySynthesizer",
        "multi_hop_abstract": "MultiHopAbstractQuerySynthesizer",
    }

    def generate_rows(
        self,
        *,
        generator: Any,
        chunks: list[Any],
        run_config: Any,
        query_type_counts: dict[str, int],
    ) -> QueryTypeSynthesisResult:
        generated_rows: list[dict[str, Any]] = []
        generated_counts = {query_type: 0 for query_type in QUERY_TYPES}
        source_map: dict[str, str] = {}

        for query_type in QUERY_TYPES:
            requested = max(int(query_type_counts.get(query_type, 0)), 0)
            if requested <= 0:
                continue

            if query_type == "single_hop_abstract":
                base_rows = _invoke_with_distribution(
                    generator,
                    chunks=chunks,
                    testset_size=requested,
                    run_config=run_config,
                    query_distribution=None,
                )
                rows = _adapt_single_hop_abstract_rows(base_rows)
                source_map[query_type] = "project_single_hop_abstract_adapter"
            else:
                class_name = self._CLASS_NAME_BY_TYPE.get(query_type)
                distribution_entry = _build_distribution_entry(class_name) if class_name else None
                distribution_payload = [distribution_entry] if distribution_entry is not None else None
                rows = _invoke_with_distribution(
                    generator,
                    chunks=chunks,
                    testset_size=requested,
                    run_config=run_config,
                    query_distribution=distribution_payload,
                )
                source_map[query_type] = class_name or "default_generator"

            normalized = _normalize_rows(rows, query_type=query_type, query_type_source=source_map[query_type])
            if len(normalized) > requested:
                normalized = normalized[:requested]
            generated_rows.extend(normalized)
            generated_counts[query_type] += len(normalized)

        return QueryTypeSynthesisResult(
            rows=generated_rows,
            requested_counts={query_type: max(int(query_type_counts.get(query_type, 0)), 0) for query_type in QUERY_TYPES},
            generated_counts=generated_counts,
            source_map=source_map,
        )
