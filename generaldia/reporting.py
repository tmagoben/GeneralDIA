"""Serializable and browser-readable reports for connected-path state tracking."""

# ruff: noqa: E501, RUF001

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .dataset import TrackedMolecularPath


def state_tracking_report_data(
    result: TrackedMolecularPath,
    *,
    coordinates: Sequence[float] | None = None,
    coordinate_label: str = "Path coordinate",
) -> dict[str, Any]:
    """Return a JSON-compatible report derived from one tracked path.

    The report retains raw and tracked energies, transition confidence, overlap
    matrices, ambiguity reasons, and every numerical threshold needed to interpret
    the tracking result. It deliberately reports no qualitative pass/fail score
    beyond the tracker's explicit ambiguity decision.
    """

    if coordinates is None:
        coordinate_values = [float(index) for index in range(len(result.raw_path))]
    else:
        if len(coordinates) != len(result.raw_path):
            raise ValueError("coordinates must contain one value per path geometry")
        coordinate_values = [float(value) for value in coordinates]
        if not all(math.isfinite(value) for value in coordinate_values):
            raise ValueError("coordinates must contain finite values")
    if not isinstance(coordinate_label, str) or not coordinate_label.strip():
        raise ValueError("coordinate_label must be a non-empty string")

    transitions = []
    for step in result.tracking.steps:
        transitions.append(
            {
                "start_index": step.start_index,
                "end_index": step.end_index,
                "permutation": list(step.permutation),
                "minimum_overlap": step.minimum_overlap,
                "assignment_margin": (
                    step.assignment_margin if math.isfinite(step.assignment_margin) else None
                ),
                "ambiguous": step.ambiguous,
                "reasons": list(step.reasons),
                "degenerate_blocks": [list(block) for block in step.degenerate_blocks],
                "near_degenerate_pairs": [
                    {"left": left, "right": right, "gap": gap}
                    for left, right, gap in step.near_degenerate_pairs
                ],
                "absolute_aligned_overlap": torch.abs(step.aligned_overlap).cpu().tolist(),
            }
        )

    energy_unit = result.raw_path.metadata.get(
        "energy_unit", result.raw_path[0].metadata.get("energy_unit", "unspecified")
    )
    return {
        "schema": "generaldia.state_tracking_report",
        "schema_version": 1,
        "path_id": result.raw_path.path_id,
        "coordinate_label": coordinate_label,
        "coordinates": coordinate_values,
        "energy_unit": str(energy_unit),
        "n_geometries": len(result.raw_path),
        "n_states": result.raw_path.n_states,
        "raw_energies": result.raw_path.energies.cpu().tolist(),
        "tracked_energies": result.tracked_energies.cpu().tolist(),
        "ambiguous_steps": list(result.ambiguous_steps),
        "settings": {
            "overlap_floor": result.settings.overlap_floor,
            "assignment_margin_floor": result.settings.assignment_margin_floor,
            "degeneracy_tolerance": result.settings.degeneracy_tolerance,
            "near_degeneracy_threshold": result.settings.near_degeneracy_threshold,
            "on_ambiguous": result.settings.on_ambiguous,
        },
        "transitions": transitions,
        "claim_boundary": (
            "The report visualizes a local continuation supported by supplied adjacent "
            "overlaps; it does not establish a unique global diabatic basis."
        ),
    }


def write_state_tracking_report(
    result: TrackedMolecularPath,
    output: str | Path,
    *,
    coordinates: Sequence[float] | None = None,
    coordinate_label: str = "Path coordinate",
) -> Path:
    """Write a self-contained interactive HTML report and return its path."""

    data = state_tracking_report_data(
        result,
        coordinates=coordinates,
        coordinate_label=coordinate_label,
    )
    payload = json.dumps(data, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_HTML_TEMPLATE.replace("__REPORT_JSON__", payload), encoding="utf-8")
    return destination


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GeneralDIA path diagnostics</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f6f8fb;
      --surface: #ffffff;
      --text: #172033;
      --muted: #667085;
      --border: #d8dee9;
      --grid: #e7ebf1;
      --accent: #3454d1;
      --danger: #b42318;
      --good: #067647;
      --heat: 52, 84, 209;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #101522;
        --surface: #171e2d;
        --text: #edf2f7;
        --muted: #aab4c4;
        --border: #354052;
        --grid: #2a3445;
        --accent: #91a7ff;
        --danger: #ff8b7f;
        --good: #6ce9a6;
        --heat: 145, 167, 255;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }
    main { width: min(1120px, calc(100% - 32px)); margin: 28px auto 48px; }
    h1, h2 { margin: 0; font-weight: 650; }
    h1 { font-size: clamp(1.45rem, 3vw, 2rem); }
    h2 { font-size: 1rem; }
    .subtitle, .claim { color: var(--muted); }
    .subtitle { margin: 6px 0 0; }
    .claim { margin: 24px 0 0; font-size: .9rem; }
    .summary {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 22px;
      margin: 18px 0 22px;
      color: var(--muted);
    }
    .summary strong { color: var(--text); font-variant-numeric: tabular-nums; }
    .status { color: var(--good); }
    .status.ambiguous { color: var(--danger); }
    .layout { display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(300px, .8fr); gap: 18px; }
    .panel {
      min-width: 0;
      padding: 18px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--surface);
    }
    .panel-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
    .panel-note { color: var(--muted); font-size: .82rem; }
    svg { display: block; width: 100%; height: auto; margin-top: 12px; }
    .axis, .grid { stroke: var(--grid); stroke-width: 1; vector-effect: non-scaling-stroke; }
    .axis { stroke: var(--border); }
    .tick, .axis-label { fill: var(--muted); font-size: 11px; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px 14px; margin-top: 10px; font-size: .82rem; color: var(--muted); }
    .legend span { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 18px; height: 3px; border-radius: 2px; background: currentColor; }
    .controls { margin-top: 16px; }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
    .transition-title { margin-top: 12px; }
    .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 12px 0 16px; }
    .metric { padding: 10px; border-radius: 8px; background: color-mix(in srgb, var(--accent) 8%, transparent); }
    .metric span { display: block; color: var(--muted); font-size: .78rem; }
    .metric strong { font-variant-numeric: tabular-nums; }
    .heatmap { display: grid; gap: 4px; width: min(100%, 360px); margin: 12px auto; }
    .heat-cell {
      aspect-ratio: 1;
      display: grid;
      place-items: center;
      min-width: 0;
      border-radius: 5px;
      color: var(--text);
      font-size: clamp(.66rem, 2vw, .82rem);
      font-variant-numeric: tabular-nums;
    }
    .mapping { margin: 12px 0 0; color: var(--muted); font-size: .88rem; }
    .mapping code { color: var(--text); }
    .reason { margin-top: 10px; color: var(--danger); font-size: .88rem; }
    .reason[hidden] { display: none; }
    .thresholds { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: .84rem; }
    .thresholds td { padding: 7px 0; border-bottom: 1px solid var(--grid); }
    .thresholds td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
    @media (max-width: 760px) {
      main { width: min(100% - 20px, 680px); margin-top: 18px; }
      .layout { grid-template-columns: 1fr; }
      .panel { padding: 14px; }
    }
  </style>
</head>
<body>
<main>
  <h1>Connected-path state diagnostics</h1>
  <p class="subtitle" id="pathTitle"></p>
  <div class="summary" aria-label="Path summary">
    <span><strong id="geometryCount"></strong> geometries</span>
    <span><strong id="stateCount"></strong> states</span>
    <span id="status" class="status"></span>
  </div>

  <div class="layout">
    <section class="panel" aria-labelledby="energyHeading">
      <div class="panel-head">
        <h2 id="energyHeading">Raw ordering and tracked character</h2>
        <span class="panel-note" id="energyUnit"></span>
      </div>
      <svg id="energyChart" viewBox="0 0 720 420" role="img" aria-label="Raw and tracked energies along the path"></svg>
      <div class="legend" id="legend" aria-label="Energy series legend"></div>
    </section>

    <section class="panel" aria-labelledby="transitionHeading">
      <div class="panel-head">
        <h2 id="transitionHeading">Transition evidence</h2>
        <span class="panel-note" id="transitionCount"></span>
      </div>
      <div class="controls">
        <label for="transitionSlider">Selected transition: <strong id="selectedTransition"></strong></label>
        <input id="transitionSlider" type="range" min="0" step="1" aria-describedby="mapping">
      </div>
      <div class="metrics">
        <div class="metric"><span>Minimum overlap</span><strong id="minimumOverlap"></strong></div>
        <div class="metric"><span>Assignment margin</span><strong id="assignmentMargin"></strong></div>
      </div>
      <h2>Absolute aligned overlap</h2>
      <div id="heatmap" class="heatmap" role="grid" aria-label="Absolute aligned overlap matrix"></div>
      <p id="mapping" class="mapping"></p>
      <p id="reason" class="reason" role="alert" hidden></p>
    </section>

    <section class="panel" aria-labelledby="confidenceHeading">
      <div class="panel-head">
        <h2 id="confidenceHeading">Tracking confidence along the path</h2>
        <span class="panel-note">Ambiguous transitions are marked ×</span>
      </div>
      <svg id="confidenceChart" viewBox="0 0 720 250" role="img" aria-label="Minimum principal overlap by transition"></svg>
    </section>

    <section class="panel" aria-labelledby="settingsHeading">
      <h2 id="settingsHeading">Recorded numerical policy</h2>
      <table class="thresholds"><tbody id="settingsTable"></tbody></table>
    </section>
  </div>
  <p class="claim" id="claim"></p>
</main>
<script id="reportData" type="application/json">__REPORT_JSON__</script>
<script>
(() => {
  "use strict";
  const data = JSON.parse(document.getElementById("reportData").textContent);
  const ns = "http://www.w3.org/2000/svg";
  const colors = ["#3454d1", "#c2417b", "#16836b", "#9a6700", "#6f42c1", "#b54708", "#087ea4", "#8e4b10"];
  const darkColors = ["#91a7ff", "#ff8ab8", "#6ce9c0", "#f5c451", "#c4a7ff", "#ffad66", "#74d4f5", "#e6ad72"];
  const palette = matchMedia("(prefers-color-scheme: dark)").matches ? darkColors : colors;
  const textColor = getComputedStyle(document.documentElement).getPropertyValue("--text").trim();
  const mutedColor = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim();
  const gridColor = getComputedStyle(document.documentElement).getPropertyValue("--grid").trim();
  const dangerColor = getComputedStyle(document.documentElement).getPropertyValue("--danger").trim();

  const svgNode = (name, attrs = {}) => {
    const node = document.createElementNS(ns, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  };
  const linePath = (points) => points.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(" ");
  const extent = (values) => [Math.min(...values), Math.max(...values)];
  const padded = ([low, high]) => {
    const span = high - low || Math.max(Math.abs(low), 1);
    return [low - span * 0.08, high + span * 0.08];
  };

  document.getElementById("pathTitle").textContent = `Path ${data.path_id}`;
  document.getElementById("geometryCount").textContent = data.n_geometries;
  document.getElementById("stateCount").textContent = data.n_states;
  document.getElementById("energyUnit").textContent = `Energy unit: ${data.energy_unit}`;
  document.getElementById("claim").textContent = data.claim_boundary;
  const status = document.getElementById("status");
  status.textContent = data.ambiguous_steps.length ? `${data.ambiguous_steps.length} ambiguous transition(s)` : "No recorded ambiguity";
  status.classList.toggle("ambiguous", Boolean(data.ambiguous_steps.length));

  function drawEnergy() {
    const svg = document.getElementById("energyChart");
    const width = 720, height = 420, left = 64, right = 22, top = 20, bottom = 54;
    const allY = data.raw_energies.flat().concat(data.tracked_energies.flat());
    const [xmin, xmax] = padded(extent(data.coordinates));
    const [ymin, ymax] = padded(extent(allY));
    const x = value => left + (value - xmin) / (xmax - xmin) * (width - left - right);
    const y = value => height - bottom - (value - ymin) / (ymax - ymin) * (height - top - bottom);
    for (let i = 0; i <= 4; i++) {
      const value = ymin + (ymax - ymin) * i / 4;
      const py = y(value);
      svg.append(svgNode("line", {x1:left, x2:width-right, y1:py, y2:py, stroke:gridColor}));
      const label = svgNode("text", {x:left-9, y:py+4, "text-anchor":"end", fill:mutedColor, "font-size":11});
      label.textContent = value.toPrecision(4);
      svg.append(label);
    }
    svg.append(svgNode("line", {x1:left, x2:left, y1:top, y2:height-bottom, stroke:gridColor}));
    svg.append(svgNode("line", {x1:left, x2:width-right, y1:height-bottom, y2:height-bottom, stroke:gridColor}));
    for (let state = 0; state < data.n_states; state++) {
      const raw = data.coordinates.map((value, i) => [x(value), y(data.raw_energies[i][state])]);
      const tracked = data.coordinates.map((value, i) => [x(value), y(data.tracked_energies[i][state])]);
      svg.append(svgNode("path", {d:linePath(raw), fill:"none", stroke:palette[state % palette.length], "stroke-width":1.4, "stroke-dasharray":"5 5", opacity:.45}));
      svg.append(svgNode("path", {d:linePath(tracked), fill:"none", stroke:palette[state % palette.length], "stroke-width":2.6}));
      tracked.forEach(([px, py]) => svg.append(svgNode("circle", {cx:px, cy:py, r:3.2, fill:palette[state % palette.length]})));
    }
    const xLabel = svgNode("text", {x:(left+width-right)/2, y:height-12, "text-anchor":"middle", fill:mutedColor, "font-size":11});
    xLabel.textContent = data.coordinate_label;
    svg.append(xLabel);
    const yLabel = svgNode("text", {x:15, y:(top+height-bottom)/2, transform:`rotate(-90 15 ${(top+height-bottom)/2})`, "text-anchor":"middle", fill:mutedColor, "font-size":11});
    yLabel.textContent = "Energy";
    svg.append(yLabel);
  }

  function drawConfidence() {
    const svg = document.getElementById("confidenceChart");
    const width = 720, height = 250, left = 64, right = 22, top = 20, bottom = 45;
    const count = data.transitions.length;
    const x = index => count === 1 ? (left + width - right) / 2 : left + index / (count - 1) * (width - left - right);
    const y = value => height - bottom - value * (height - top - bottom);
    [0, .5, 1].forEach(value => {
      svg.append(svgNode("line", {x1:left, x2:width-right, y1:y(value), y2:y(value), stroke:gridColor}));
      const label = svgNode("text", {x:left-9, y:y(value)+4, "text-anchor":"end", fill:mutedColor, "font-size":11});
      label.textContent = value.toFixed(1);
      svg.append(label);
    });
    const points = data.transitions.map((step, i) => [x(i), y(step.minimum_overlap)]);
    svg.append(svgNode("path", {d:linePath(points), fill:"none", stroke:palette[0], "stroke-width":2.5}));
    data.transitions.forEach((step, i) => {
      const px = x(i), py = y(step.minimum_overlap);
      svg.append(svgNode("circle", {cx:px, cy:py, r:4, fill:step.ambiguous ? dangerColor : palette[0]}));
      if (step.ambiguous) {
        const mark = svgNode("text", {x:px, y:py-9, "text-anchor":"middle", fill:dangerColor, "font-size":15});
        mark.textContent = "×";
        svg.append(mark);
      }
    });
    const label = svgNode("text", {x:(left+width-right)/2, y:height-10, "text-anchor":"middle", fill:mutedColor, "font-size":11});
    label.textContent = "Transition index";
    svg.append(label);
  }

  function updateTransition(index) {
    const step = data.transitions[index];
    document.getElementById("selectedTransition").textContent = `${step.start_index} → ${step.end_index}`;
    document.getElementById("minimumOverlap").textContent = step.minimum_overlap.toFixed(6);
    document.getElementById("assignmentMargin").textContent = step.assignment_margin === null ? "∞" : step.assignment_margin.toExponential(3);
    document.getElementById("mapping").innerHTML = `Tracked → raw state mapping: <code>${step.permutation.map((raw, tracked) => `S${tracked}→S${raw}`).join(", ")}</code>`;
    const reason = document.getElementById("reason");
    reason.hidden = !step.reasons.length;
    reason.textContent = step.reasons.join("; ");
    const heatmap = document.getElementById("heatmap");
    heatmap.replaceChildren();
    heatmap.style.gridTemplateColumns = `repeat(${data.n_states}, minmax(0, 1fr))`;
    step.absolute_aligned_overlap.forEach((row, rowIndex) => row.forEach((value, columnIndex) => {
      const cell = document.createElement("div");
      cell.className = "heat-cell";
      cell.setAttribute("role", "gridcell");
      cell.setAttribute("aria-label", `row ${rowIndex}, column ${columnIndex}, overlap ${value.toFixed(6)}`);
      cell.style.background = `rgba(var(--heat), ${(.06 + .78 * value).toFixed(3)})`;
      cell.textContent = value.toFixed(3);
      heatmap.append(cell);
    }));
  }

  const legend = document.getElementById("legend");
  for (let state = 0; state < data.n_states; state++) {
    const item = document.createElement("span");
    item.style.color = palette[state % palette.length];
    item.innerHTML = `<i class="swatch"></i><span style="color:${textColor}">S${state} tracked; dashed = raw rank</span>`;
    legend.append(item);
  }
  const slider = document.getElementById("transitionSlider");
  slider.max = String(data.transitions.length - 1);
  slider.value = "0";
  slider.addEventListener("input", () => updateTransition(Number(slider.value)));
  document.getElementById("transitionCount").textContent = `${data.transitions.length} total`;

  const labels = {
    overlap_floor: "Overlap floor",
    assignment_margin_floor: "Assignment-margin floor",
    degeneracy_tolerance: "Degeneracy tolerance",
    near_degeneracy_threshold: "Near-degeneracy threshold",
    on_ambiguous: "Ambiguity policy"
  };
  const tbody = document.getElementById("settingsTable");
  Object.entries(data.settings).forEach(([key, value]) => {
    const row = document.createElement("tr");
    const name = document.createElement("td");
    const setting = document.createElement("td");
    name.textContent = labels[key] || key;
    setting.textContent = value === null ? "not set" : String(value);
    row.append(name, setting);
    tbody.append(row);
  });
  drawEnergy();
  drawConfidence();
  updateTransition(0);
})();
</script>
</body>
</html>
"""
