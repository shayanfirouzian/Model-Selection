#!/usr/bin/env python3
"""Build one Copeland-ranked reference response per question from pairwise data.

No model inference is performed. The script reads the project's ProlificAI
pairwise-comparison file, normalizes answer whitespace, constructs the partial
preference graph for each exact question, and emits the uniquely top-ranked
answer unless the top score is tied or the top answer belongs to a directed
cycle. The rule follows the procedure documented in docs/methods_social_reasoning.md.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def display_path(path: Path) -> str:
    """Return a repository-relative path when the file is inside the project."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


DEFAULT_INPUT = ROOT / "data/social_reasoning/ProlificAI.jsonl"
DEFAULT_OUTPUT = ROOT / "data/social_reasoning/reference_responses/most_preferred.jsonl"
DEFAULT_FLAGGED = ROOT / "results/social_reasoning/copeland_single_reference_393/copeland_flagged_questions.jsonl"


def normalize(text: str) -> str:
    return " ".join(str(text or "").replace("\r\n", "\n").replace("\r", "\n").split())


def cyclic_nodes(edges: dict[str, set[str]]) -> set[str]:
    """Return nodes belonging to a strongly connected component of size > 1."""
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    discovery: dict[str, int] = {}
    low: dict[str, int] = {}
    cyclic: set[str] = set()

    def visit(v: str) -> None:
        nonlocal index
        discovery[v] = low[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w in edges.get(v, set()):
            if w not in discovery:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], discovery[w])
        if low[v] == discovery[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.append(w)
                if w == v:
                    break
            if len(component) > 1:
                cyclic.update(component)

    nodes = set(edges)
    for targets in edges.values():
        nodes.update(targets)
    for node in sorted(nodes):
        if node not in discovery:
            visit(node)
    return cyclic


def build(input_path: Path):
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with input_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question = str(row.get("question", "")).strip()
            chosen = normalize(row.get("chosen", ""))
            rejected = normalize(row.get("rejected", ""))
            if not question or not chosen or not rejected:
                raise ValueError(f"Missing question/chosen/rejected at source line {line_no}")
            if chosen == rejected:
                raise ValueError(f"Chosen and rejected normalize to the same answer at line {line_no}")
            groups[question].append((chosen, rejected))

    winners, flagged = [], []
    contradiction_count = 0
    repeated_comparison_count = 0
    for question, pairs in groups.items():
        answers = []
        seen = set()
        for chosen, rejected in pairs:
            for answer in (chosen, rejected):
                if answer not in seen:
                    seen.add(answer)
                    answers.append(answer)
        directed_pairs: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
        for chosen, rejected in pairs:
            directed_pairs[tuple(sorted((chosen, rejected)))].add((chosen, rejected))
        contradictions = [key for key, directions in directed_pairs.items() if len(directions) > 1]
        if contradictions:
            contradiction_count += len(contradictions)
            raise ValueError(f"Contradictory preference direction(s) for question {question[:100]!r}")
        repeated_comparison_count += sum(1 for key in directed_pairs if sum(1 for pair in pairs if tuple(sorted(pair)) == key) > 1)

        edges: dict[str, set[str]] = {answer: set() for answer in answers}
        scores = {answer: 0 for answer in answers}
        matrix_edges = set()
        for chosen, rejected in pairs:
            edge_key = (chosen, rejected)
            if edge_key in matrix_edges:
                continue
            matrix_edges.add(edge_key)
            edges[chosen].add(rejected)
            scores[chosen] += 1
            scores[rejected] -= 1
        top_score = max(scores.values())
        top = [answer for answer, score in scores.items() if score == top_score]
        cyclic = cyclic_nodes(edges)
        if len(top) != 1:
            flagged.append({"question": question, "reason": "top_score_tie", "top_candidates": top, "copeland_scores": scores})
        elif top[0] in cyclic:
            flagged.append({"question": question, "reason": "top_answer_in_cycle", "top_candidates": top, "copeland_scores": scores})
        else:
            winners.append({"question": question, "answer": top[0]})
    try:
        portable_input_path = input_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        portable_input_path = input_path.name
    audit = {
        "source_path": portable_input_path,
        "source_rows": sum(len(v) for v in groups.values()),
        "distinct_exact_questions": len(groups),
        "group_sizes": sorted({len(v) for v in groups.values()}),
        "winners_written": len(winners),
        "flagged_questions": len(flagged),
        "contradictory_unordered_pairs": contradiction_count,
        "repeated_unordered_pair_keys": repeated_comparison_count,
        "method": "Copeland wins-minus-losses on a partial pairwise-comparison graph; flag non-unique top score or top answer in directed cycle",
    }
    return winners, flagged, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--flagged", type=Path, default=DEFAULT_FLAGGED)
    args = parser.parse_args()
    winners, flagged, audit = build(args.input)
    for path in (args.output, args.flagged):
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in winners:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with args.flagged.open("w", encoding="utf-8") as handle:
        for item in flagged:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    audit_path = args.flagged.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    def display_path(path: Path) -> str:
        try:
            return path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            return path.name
    print(f"Winner file: {display_path(args.output)}")
    print(f"Flagged-question file: {display_path(args.flagged)}")

if __name__ == "__main__":
    main()
