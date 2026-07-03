#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from pathlib import Path
from statistics import mean

from Bio import SeqIO

DEFAULT_EPITOPE_LABEL_PATTERNS = (
    r"\bgcn4_v4\b",
    r"\bgcn4 peptide\b",
    r"\bha tag\b",
    r"\bha epitope\b",
    r"\balfa tag\b",
    r"\balfa epitope\b",
)


def nt_interval_to_aa_bounds(start0: int, end0: int) -> tuple[float, float]:
    start1 = start0 + 1
    end1 = end0
    return (start1 / 3.0, end1 / 3.0)


def get_longest_cds_aa_length(record) -> float:
    cds_features = [f for f in record.features if f.type == "CDS"]
    if not cds_features:
        raise ValueError(f"{record.id}: no CDS feature found")

    longest_cds = max(cds_features, key=lambda f: len(f.location))
    return len(longest_cds.location) / 3.0


def extract_epitope_features_aa(record, epitope_label_hint: str | None) -> list[dict]:
    """Extract individual epitope annotations from common label styles."""
    hint = epitope_label_hint.strip().lower() if epitope_label_hint else ""
    epitopes: list[dict] = []
    pattern: re.Pattern[str] | None = None
    auto_patterns = [re.compile(p) for p in DEFAULT_EPITOPE_LABEL_PATTERNS]

    if hint:
        if hint in {"ha", "alfa", "gcn4"}:
            pattern = re.compile(rf"\b{re.escape(hint)}\b")
        else:
            pattern = re.compile(re.escape(hint))

    for feature in record.features:
        fields = []
        for key in ("label", "note", "product"):
            fields.extend(feature.qualifiers.get(key, []))
        text = " ".join(str(v) for v in fields).strip().lower()
        if not text:
            continue
        if pattern:
            matched = bool(pattern.search(text))
        else:
            matched = any(p.search(text) for p in auto_patterns)

        if matched:
            start0 = int(feature.location.start)
            end0 = int(feature.location.end)
            start_aa, end_aa = nt_interval_to_aa_bounds(start0, end0)
            epitopes.append({"start_aa": start_aa, "end_aa": end_aa})

    return epitopes


def compute_metrics(orf_aa: float, epitopes_aa: list[dict]) -> dict:
    if len(epitopes_aa) == 0:
        raise ValueError("No epitope annotations found")

    epitope_complete_pos_aa = sorted(e["end_aa"] for e in epitopes_aa)
    epitope_starts_aa = [e["start_aa"] for e in epitopes_aa]
    epitope_ends_aa = [e["end_aa"] for e in epitopes_aa]

    tag_start_aa = min(epitope_starts_aa)
    tag_end_aa = max(epitope_ends_aa)
    tag_span_aa = tag_end_aa - tag_start_aa
    if tag_span_aa <= 0:
        raise ValueError("Invalid epitope tag span")

    # Expected detectable SunTag fraction while a ribosome traverses the tag span.
    # Uses epitope completion positions (3' / end coordinates), since an epitope
    # contributes only once fully translated. Uniform end-position spacing across
    # the tag span yields ~0.5 naturally.
    suntag_effective_fraction = mean(
        (tag_end_aa - p) / tag_span_aa for p in epitope_complete_pos_aa
    )

    # For correction-factor weighting, treat the "tag-side" region as ORF start
    # through the last epitope end (not only first-epitope..last-epitope span).
    tag_region_aa = tag_end_aa
    renilla_or_non_tag_aa = max(0.0, orf_aa - tag_region_aa)
    correction_factor = (
        (suntag_effective_fraction * tag_region_aa) + (1.0 * renilla_or_non_tag_aa)
    ) / orf_aa
    apparent_orf_aa = correction_factor * orf_aa

    return {
        "orf_aa": orf_aa,
        "n_epitopes": len(epitope_complete_pos_aa),
        "tag_start_aa": tag_start_aa,
        "tag_end_aa": tag_end_aa,
        "tag_span_aa": tag_span_aa,
        "tag_region_aa": tag_region_aa,
        "mean_epitope_completion_pos_aa": mean(epitope_complete_pos_aa),
        "suntag_effective_fraction": suntag_effective_fraction,
        "correction_factor": correction_factor,
        "apparent_orf_aa": apparent_orf_aa,
    }


def summarize_file(path: Path, epitope_label_hint: str | None) -> dict:
    record = SeqIO.read(path, "genbank")
    orf_aa = get_longest_cds_aa_length(record)
    epitopes_aa = extract_epitope_features_aa(record, epitope_label_hint)
    stats = compute_metrics(orf_aa, epitopes_aa)

    return {
        "file": path.name,
        "record_id": record.id,
        "epitope_completion_positions_aa": sorted(e["end_aa"] for e in epitopes_aa),
        **stats,
    }


def print_result(result: dict) -> None:
    print(f"\nFile: {result['file']}")
    print(f"Record ID: {result['record_id']}")
    print(f"ORF length (aa): {result['orf_aa']:.3f}")
    print(f"Epitopes counted: {result['n_epitopes']}")
    print(
        "Epitope tag span (aa): "
        f"{result['tag_start_aa']:.3f} to {result['tag_end_aa']:.3f} "
        f"(length {result['tag_span_aa']:.3f})"
    )
    print(
        "Tag-side region used in correction (aa 1 to last epitope end): "
        f"{result['tag_region_aa']:.3f}"
    )
    print(
        "Mean epitope completion position (aa; 3' end of each epitope): "
        f"{result['mean_epitope_completion_pos_aa']:.3f}"
    )
    print(
        "Computed SunTag occupancy term (old model's 0.5 analog): "
        f"{result['suntag_effective_fraction']:.5f}"
    )
    print(f"Computed correction factor: {result['correction_factor']:.5f}")
    print(
        "Apparent ORF size (aa) = correction_factor * ORF length: "
        f"{result['apparent_orf_aa']:.3f}"
    )


def print_comparison(a: dict, b: dict) -> None:
    raw_diff_pct = ((b["orf_aa"] - a["orf_aa"]) / a["orf_aa"]) * 100.0
    apparent_orf_diff_pct = (
        (b["apparent_orf_aa"] - a["apparent_orf_aa"]) / a["apparent_orf_aa"]
    ) * 100.0

    print("\nComparison")
    print(f"Reference file: {a['file']}")
    print(f"Comparison file: {b['file']}")
    print(f"Raw ORF difference (%): {raw_diff_pct:.3f}")
    print(f"Apparent ORF-size difference (%): {apparent_orf_diff_pct:.3f}")


def prompt_for_file_paths() -> list[Path]:
    print("How many GenBank files do you want to compare? (min 2)")
    count_text = input("> ").strip()
    count = int(count_text)
    if count < 2:
        raise ValueError("Need at least 2 files for comparison")

    paths: list[Path] = []
    for i in range(1, count + 1):
        print(f"Paste path to GenBank file {i}:")
        raw = input("> ").strip().strip('"').strip("'")
        if not raw:
            raise ValueError("All file paths are required")
        paths.append(Path(raw))
    return paths


def prompt_for_epitope_hint(path: Path) -> str | None:
    print(
        f"Epitopes for {path.name}: enter label keyword "
        "(e.g., gcn4, gcn4 peptide, gcn4_v4, ha, alfa) or press Enter to auto-detect:"
    )
    hint = input("> ").strip().strip('"').strip("'")
    return hint if hint else None


def main(argv: list[str]) -> int:
    if len(argv) == 0:
        paths = prompt_for_file_paths()
    else:
        paths = [Path(arg) for arg in argv]

    results = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        hint = prompt_for_epitope_hint(path) if len(argv) == 0 else None
        result = summarize_file(path, hint)
        print_result(result)
        results.append(result)

        if result["n_epitopes"] != 18:
            print(
                f"WARNING: expected 18 epitopes, found {result['n_epitopes']} "
                f"in {result['file']}"
            )

    if len(results) >= 2:
        for result in results[1:]:
            print_comparison(results[0], result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))