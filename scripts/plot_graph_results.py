#!/usr/bin/env python3
"""Render a dependency-free SVG summary of the hybrid graph results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


COLORS = {"precision": "#2563eb", "recall": "#16a34a", "f1": "#ea580c"}
LABELS = {
    "confirmed_relationship": "Confirmed relationship",
    "counter_enhanced_gae_alert": "Counter-enhanced GAE",
    "any_graph_signal": "Combined review signal",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default="outputs/graph_evaluation/hybrid_alerts/metrics.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/graph_evaluation/figures/graph_detector_results.svg",
    )
    args = parser.parse_args()
    report = json.loads(Path(args.metrics).read_text())
    metrics = report["metrics"]

    width, height = 1600, 1000
    left, top, chart_width, chart_height = 150, 245, 1320, 500
    groups = list(LABELS)
    series = ["precision", "recall", "f1"]
    group_width = chart_width / len(groups)
    bar_width = 92
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1600" height="1000" fill="#f8fafc"/>',
        '<text x="800" y="72" text-anchor="middle" font-family="Arial" font-size="42" font-weight="700" fill="#0f172a">Long-Term Graph Detector Results</text>',
        '<text x="800" y="118" text-anchor="middle" font-family="Arial" font-size="24" fill="#475569">Retrospective evaluation: Days 17–30</text>',
    ]

    for index, name in enumerate(series):
        x = 485 + index * 230
        svg.extend([
            f'<rect x="{x}" y="157" width="28" height="28" rx="4" fill="{COLORS[name]}"/>',
            f'<text x="{x + 40}" y="179" font-family="Arial" font-size="22" fill="#334155">{name.title()}</text>',
        ])

    for tick in range(0, 101, 20):
        y = top + chart_height - chart_height * tick / 100
        svg.extend([
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#cbd5e1" stroke-width="1"/>',
            f'<text x="{left - 22}" y="{y + 8:.1f}" text-anchor="end" font-family="Arial" font-size="20" fill="#64748b">{tick}%</text>',
        ])

    for group_index, key in enumerate(groups):
        center = left + group_width * (group_index + 0.5)
        for series_index, name in enumerate(series):
            value = float(metrics[key][name])
            x = center + (series_index - 1) * (bar_width + 18) - bar_width / 2
            bar_height = chart_height * value
            y = top + chart_height - bar_height
            svg.extend([
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="8" fill="{COLORS[name]}"/>',
                f'<text x="{x + bar_width / 2:.1f}" y="{max(y - 12, top + 20):.1f}" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700" fill="#0f172a">{value * 100:.2f}%</text>',
            ])
        svg.append(
            f'<text x="{center:.1f}" y="790" text-anchor="middle" font-family="Arial" font-size="23" font-weight="600" fill="#1e293b">{LABELS[key]}</text>'
        )

    confirmed = metrics["confirmed_relationship"]
    svg.extend([
        '<rect x="150" y="840" width="1320" height="110" rx="18" fill="#e0f2fe" stroke="#7dd3fc"/>',
        f'<text x="190" y="885" font-family="Arial" font-size="25" font-weight="700" fill="#0c4a6e">Primary high-confidence result: F1 {confirmed["f1"] * 100:.2f}%</text>',
        f'<text x="190" y="925" font-family="Arial" font-size="21" fill="#075985">TP {confirmed["tp"]}  •  FP {confirmed["fp"]}  •  FN {confirmed["fn"]}  •  TN {confirmed["tn"]:,}</text>',
        '<text x="1415" y="907" text-anchor="end" font-family="Arial" font-size="18" fill="#475569">84.06% is the supervised recurrence result, not GAE F1.</text>',
        '</svg>',
    ])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg) + "\n")
    print(output)


if __name__ == "__main__":
    main()
