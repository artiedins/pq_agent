const SVG_NS = "http://www.w3.org/2000/svg";

const LAB_META = [
  { key: "openai", label: "OpenAI", cssVar: "--lab-openai" },
  { key: "anthropic", label: "Anthropic", cssVar: "--lab-anthropic" },
  { key: "google", label: "Google", cssVar: "--lab-google" },
  { key: "meta", label: "Meta", cssVar: "--lab-meta" },
  { key: "xai", label: "xAI", cssVar: "--lab-xai" },
  { key: "deepseek", label: "DeepSeek", cssVar: "--lab-deepseek" },
  { key: "qwen", label: "Qwen", cssVar: "--lab-qwen" },
  { key: "other", label: "Other Labs", cssVar: "--lab-other" },
];

const TIER_INFO = {
  govcloud: {
    short: "AWS GovCloud",
    badge: "AWS GovCloud · Export Control",
    scatterTag: "GovCloud",
    tableText: "AWS GovCloud (Export-Controlled)",
    tooltipText: "Honeywell Aerospace Approved · AWS GovCloud (Export Control)",
  },
  onprem_open: {
    short: "On-Prem Open",
    badge: "On-Prem Open-Weight · Non-Chinese",
    scatterTag: "On-Prem Open",
    tableText: "On-Prem Open-Weight (Non-Chinese)",
    tooltipText: "Honeywell Eligible · On-Prem Open-Weight (Non-Chinese)",
  },
  standard: {
    short: "Commercial / Other",
    badge: "",
    scatterTag: "",
    tableText: "Standard (Not GovCloud / On-Prem)",
    tooltipText: "",
  },
};

const state = {
  data: null,
  tierFilter: "all",
  labFilter: null,
  scale: "log",
  showFrontier: true,
  hoveredSlug: null,
};

function svgEl(tag, attrs = {}, text = null) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) {
      el.setAttribute(k, String(v));
    }
  }
  if (text !== null) {
    el.textContent = text;
  }
  return el;
}

function labColor(labKey) {
  return `var(--lab-${labKey})`;
}

function formatCost(cost) {
  if (cost === 0) return "$0.00";
  if (cost < 1) return `$${cost.toFixed(2)}`;
  if (cost < 10) return `$${cost.toFixed(2)}`;
  return `$${cost.toFixed(2)}`;
}

function isModelHighlighted(model) {
  if (state.labFilter && model.labKey !== state.labFilter) {
    return false;
  }
  if (state.tierFilter === "eligible") {
    return model.tier === "govcloud" || model.tier === "onprem_open";
  }
  if (state.tierFilter === "govcloud") {
    return model.tier === "govcloud";
  }
  if (state.tierFilter === "onprem_open") {
    return model.tier === "onprem_open";
  }
  return true;
}

function createLabMark(labKey, x, y, size = 14) {
  const g = svgEl("g", {
    transform: `translate(${x}, ${y})`,
    "aria-hidden": "true",
  });
  const s = size;
  const c = labColor(labKey);

  if (labKey === "openai") {
    g.appendChild(svgEl("circle", { cx: s / 2, cy: s / 2, r: s * 0.44, fill: "none", stroke: c, "stroke-width": 1.7 }));
    g.appendChild(svgEl("path", {
      d: `M ${s * 0.5} ${s * 0.14} L ${s * 0.78} ${s * 0.68} L ${s * 0.22} ${s * 0.68} Z`,
      fill: "none",
      stroke: c,
      "stroke-width": 1.35,
      "stroke-linejoin": "round",
    }));
  } else if (labKey === "anthropic") {
    g.appendChild(svgEl("path", {
      d: `M ${s * 0.2} ${s * 0.85} L ${s * 0.48} ${s * 0.15} L ${s * 0.64} ${s * 0.15} L ${s * 0.88} ${s * 0.85}`,
      fill: "none",
      stroke: c,
      "stroke-width": 2,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
    }));
    g.appendChild(svgEl("line", {
      x1: s * 0.33, y1: s * 0.62, x2: s * 0.74, y2: s * 0.62,
      stroke: c, "stroke-width": 1.7, "stroke-linecap": "round",
    }));
  } else if (labKey === "google") {
    g.appendChild(svgEl("path", {
      d: `M ${s * 0.5} ${s * 0.06} Q ${s * 0.5} ${s * 0.5} ${s * 0.94} ${s * 0.5} Q ${s * 0.5} ${s * 0.5} ${s * 0.5} ${s * 0.94} Q ${s * 0.5} ${s * 0.5} ${s * 0.06} ${s * 0.5} Q ${s * 0.5} ${s * 0.5} ${s * 0.5} ${s * 0.06} Z`,
      fill: c,
    }));
  } else if (labKey === "meta") {
    g.appendChild(svgEl("path", {
      d: `M ${s * 0.14} ${s * 0.56} C ${s * 0.14} ${s * 0.26}, ${s * 0.36} ${s * 0.26}, ${s * 0.5} ${s * 0.5} C ${s * 0.64} ${s * 0.74}, ${s * 0.86} ${s * 0.74}, ${s * 0.86} ${s * 0.44} C ${s * 0.86} ${s * 0.24}, ${s * 0.64} ${s * 0.24}, ${s * 0.5} ${s * 0.5} C ${s * 0.36} ${s * 0.76}, ${s * 0.14} ${s * 0.76}, ${s * 0.14} ${s * 0.56} Z`,
      fill: "none",
      stroke: c,
      "stroke-width": 1.8,
      "stroke-linecap": "round",
    }));
  } else if (labKey === "xai") {
    g.appendChild(svgEl("path", {
      d: `M ${s * 0.2} ${s * 0.2} L ${s * 0.8} ${s * 0.8} M ${s * 0.8} ${s * 0.2} L ${s * 0.2} ${s * 0.8}`,
      stroke: c,
      "stroke-width": 2.1,
      "stroke-linecap": "round",
    }));
  } else if (labKey === "deepseek") {
    g.appendChild(svgEl("path", {
      d: `M ${s * 0.14} ${s * 0.58} Q ${s * 0.35} ${s * 0.16} ${s * 0.72} ${s * 0.42} Q ${s * 0.88} ${s * 0.54} ${s * 0.7} ${s * 0.74} Q ${s * 0.36} ${s * 0.9} ${s * 0.14} ${s * 0.58} Z`,
      fill: c,
    }));
  } else if (labKey === "qwen") {
    g.appendChild(svgEl("polygon", {
      points: `${s * 0.5},${s * 0.1} ${s * 0.88},${s * 0.32} ${s * 0.88},${s * 0.68} ${s * 0.5},${s * 0.9} ${s * 0.12},${s * 0.68} ${s * 0.12},${s * 0.32}`,
      fill: "none",
      stroke: c,
      "stroke-width": 1.8,
      "stroke-linejoin": "round",
    }));
    g.appendChild(svgEl("circle", { cx: s * 0.5, cy: s * 0.5, r: s * 0.16, fill: c }));
  } else {
    g.appendChild(svgEl("rect", {
      x: s * 0.18, y: s * 0.18, width: s * 0.64, height: s * 0.64, rx: 3,
      fill: "none", stroke: c, "stroke-width": 1.7,
    }));
    g.appendChild(svgEl("circle", { cx: s * 0.5, cy: s * 0.5, r: s * 0.14, fill: c }));
  }
  return g;
}

function createShieldIcon(x, y, color) {
  return svgEl("path", {
    d: "M6 1 10.6 2.9v3.2c0 2.7-1.9 4.6-4.6 5.5C3.3 10.7 1.4 8.8 1.4 6.1V2.9L6 1Z",
    transform: `translate(${x}, ${y})`,
    fill: "none",
    stroke: color,
    "stroke-width": 1.45,
    "stroke-linejoin": "round",
    "aria-hidden": "true",
  });
}

function createCubeIcon(x, y, color) {
  const g = svgEl("g", { transform: `translate(${x}, ${y})`, "aria-hidden": "true" });
  g.appendChild(svgEl("path", {
    d: "M6 1.2 10.6 3.7v4.8L6 11 1.4 8.5V3.7L6 1.2Z",
    fill: "none",
    stroke: color,
    "stroke-width": 1.35,
    "stroke-linejoin": "round",
  }));
  g.appendChild(svgEl("path", {
    d: "M1.4 3.7 6 6.1l4.6-2.4M6 6.1V11",
    fill: "none",
    stroke: color,
    "stroke-width": 1.2,
  }));
  return g;
}

function renderBarChart() {
  const container = document.getElementById("bar-chart-stage");
  const models = state.data.models;

  const width = 1040;
  const topPad = 42;
  const bottomPad = 46;
  const rowH = 34;
  const barH = 20;
  const chartLeft = 248;
  const chartRight = 986;
  const chartW = chartRight - chartLeft;
  const height = topPad + models.length * rowH + bottomPad;

  const svg = svgEl("svg", {
    class: "chart-svg",
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-labelledby": "bar-svg-title bar-svg-desc",
  });

  svg.appendChild(svgEl("title", { id: "bar-svg-title" }, "Composite Benchmark Score Horizontal Bar Chart"));
  svg.appendChild(svgEl("desc", { id: "bar-svg-desc" },
    "Horizontal bar chart ranking 27 AI models from highest to lowest equal-weight mean of ADEV Bench, TB Science, and Trivia Bench on a 0 to 100 zero-baseline scale, highlighting two AWS GovCloud models and three on-prem open-weight non-Chinese models."
  ));

  const defs = svgEl("defs");
  const hatch = svgEl("pattern", {
    id: "open-weight-hatch",
    width: 7,
    height: 7,
    patternUnits: "userSpaceOnUse",
    patternTransform: "rotate(45)",
  });
  hatch.appendChild(svgEl("line", {
    x1: 0, y1: 0, x2: 0, y2: 7,
    stroke: "#ffffff",
    "stroke-opacity": 0.42,
    "stroke-width": 2.4,
  }));
  defs.appendChild(hatch);
  svg.appendChild(defs);

  const ticks = [0, 20, 40, 60, 80, 100];
  const gridGroup = svgEl("g", { "aria-hidden": "true" });

  for (const t of ticks) {
    const x = chartLeft + (t / 100) * chartW;
    gridGroup.appendChild(svgEl("line", {
      x1: x, y1: topPad - 6, x2: x, y2: height - bottomPad + 6,
      stroke: t === 0 ? "var(--axis)" : "var(--grid)",
      "stroke-width": t === 0 ? 1.5 : 1,
      "stroke-dasharray": t === 0 ? null : "3 3",
    }));

    gridGroup.appendChild(svgEl("text", {
      x, y: topPad - 15,
      "text-anchor": "middle",
      fill: "var(--axis)",
      "font-size": 12,
      "font-weight": 550,
    }, String(t)));

    gridGroup.appendChild(svgEl("text", {
      x, y: height - bottomPad + 22,
      "text-anchor": "middle",
      fill: "var(--axis)",
      "font-size": 12,
      "font-weight": 550,
    }, String(t)));
  }

  gridGroup.appendChild(svgEl("text", {
    x: chartLeft + chartW / 2,
    y: height - 6,
    "text-anchor": "middle",
    fill: "var(--text-muted)",
    "font-size": 12,
    "font-weight": 600,
  }, "Composite Benchmark Score (Equal-Weight Mean, 0–100)"));

  svg.appendChild(gridGroup);

  const rowsGroup = svgEl("g");

  models.forEach((m, idx) => {
    const y = topPad + idx * rowH;
    const barY = y + (rowH - barH) / 2;
    const barW = Math.max(2, (m.composite / 100) * chartW);
    const isGov = m.tier === "govcloud";
    const isOpen = m.tier === "onprem_open";
    const highlighted = isModelHighlighted(m);

    const rowG = svgEl("g", {
      class: `bar-row${highlighted ? "" : " is-dimmed"}${state.hoveredSlug === m.slug ? " is-hovered" : ""}`,
      "data-slug": m.slug,
      tabindex: "0",
      role: "button",
      "aria-label": `Rank ${m.rank}, ${m.name} by ${m.lab}, composite score ${m.composite.toFixed(1)}, task cost ${formatCost(m.cost)}${m.tier !== "standard" ? ", " + TIER_INFO[m.tier].tableText : ""}`,
    });

    if (isGov || isOpen) {
      const bandFill = isGov ? "var(--tier-govcloud-bg)" : "var(--tier-onprem-bg)";
      const accent = isGov ? "var(--tier-govcloud)" : "var(--tier-onprem)";
      rowG.appendChild(svgEl("rect", {
        x: 0, y: y + 1, width: width, height: rowH - 2, rx: 6,
        fill: bandFill,
      }));
      rowG.appendChild(svgEl("rect", {
        x: 0, y: y + 2, width: 4, height: rowH - 4, rx: 2,
        fill: accent,
      }));
    }

    rowG.appendChild(svgEl("rect", {
      class: "row-hover-bg",
      x: 0, y: y + 1, width: width, height: rowH - 2, rx: 6,
      fill: "var(--surface-subtle)",
      "fill-opacity": 0,
    }));

    rowG.appendChild(svgEl("rect", {
      class: "row-focus-box",
      x: 1, y: y + 1, width: width - 2, height: rowH - 2, rx: 6,
      fill: "none",
      stroke: "var(--focus-ring)",
      "stroke-opacity": 0,
      "stroke-width": 2,
    }));

    rowG.appendChild(svgEl("text", {
      x: 32,
      y: y + rowH / 2 + 4.5,
      "text-anchor": "end",
      fill: isGov ? "var(--tier-govcloud)" : isOpen ? "var(--tier-onprem)" : "var(--text-muted)",
      "font-size": 12,
      "font-weight": isGov || isOpen ? 700 : 500,
    }, String(m.rank)));

    rowG.appendChild(createLabMark(m.labKey, 42, y + (rowH - 14) / 2, 14));

    rowG.appendChild(svgEl("text", {
      x: 64,
      y: y + rowH / 2 + 4.5,
      fill: "var(--text)",
      "font-size": 13.2,
      "font-weight": isGov || isOpen || m.rank <= 3 ? 650 : 500,
    }, m.name));

    rowG.appendChild(svgEl("rect", {
      x: chartLeft,
      y: barY,
      width: barW,
      height: barH,
      rx: 4,
      fill: labColor(m.labKey),
      stroke: isGov ? "var(--tier-govcloud)" : isOpen ? "var(--tier-onprem)" : "none",
      "stroke-width": isGov || isOpen ? 1.5 : 0,
    }));

    if (isOpen) {
      rowG.appendChild(svgEl("rect", {
        x: chartLeft,
        y: barY,
        width: barW,
        height: barH,
        rx: 4,
        fill: "url(#open-weight-hatch)",
        "pointer-events": "none",
      }));
    }

    const scoreX = chartLeft + barW + 9;
    rowG.appendChild(svgEl("text", {
      x: scoreX,
      y: y + rowH / 2 + 4.5,
      fill: "var(--text)",
      "font-size": 13,
      "font-weight": isGov || isOpen || m.rank <= 3 ? 700 : 600,
    }, m.composite.toFixed(1)));

    if (isGov || isOpen) {
      const badgeX = scoreX + 42;
      const badgeLabel = TIER_INFO[m.tier].badge;
      const badgeW = isGov ? 208 : 244;
      const accent = isGov ? "var(--tier-govcloud)" : "var(--tier-onprem)";

      rowG.appendChild(svgEl("rect", {
        x: badgeX,
        y: y + 6,
        width: badgeW,
        height: 22,
        rx: 11,
        fill: "var(--surface)",
        stroke: accent,
        "stroke-width": 1.25,
      }));

      if (isGov) {
        rowG.appendChild(createShieldIcon(badgeX + 8, y + 10.5, accent));
      } else {
        rowG.appendChild(createCubeIcon(badgeX + 8, y + 10.5, accent));
      }

      rowG.appendChild(svgEl("text", {
        x: badgeX + 24,
        y: y + 21,
        fill: accent,
        "font-size": 11.2,
        "font-weight": 680,
      }, badgeLabel));
    }

    bindInteractiveEvents(rowG, m);
    rowsGroup.appendChild(rowG);
  });

  svg.appendChild(rowsGroup);
  container.replaceChildren(svg);
}

function estimateLabelWidth(text, isPill) {
  return text.length * (isPill ? 6.6 : 6.1) + (isPill ? 18 : 6);
}

function rectsOverlap(a, b, pad = 3) {
  return !(
    a.x + a.w + pad <= b.x ||
    b.x + b.w + pad <= a.x ||
    a.y + a.h + pad <= b.y ||
    b.y + b.h + pad <= a.y
  );
}

function renderScatterChart() {
  const container = document.getElementById("scatter-chart-stage");
  const models = state.data.models;
  const frontierSet = new Set(state.data.frontier);
  const isLog = state.scale === "log";

  const width = 1040;
  const height = 560;
  const margin = { top: 30, right: 34, bottom: 64, left: 74 };
  const plotTop = margin.top;
  const plotBottom = height - margin.bottom;
  const plotLeft = margin.left;
  const plotRight = width - margin.right;
  const plotH = plotBottom - plotTop;

  const yMin = 18;
  const yMax = 94;
  const yScale = (score) => plotBottom - ((score - yMin) / (yMax - yMin)) * plotH;

  // Every whitelisted model now reports a positive price, so the log axis runs the full
  // plot width with a small inset on each side instead of reserving a $0 break band.
  const logStart = plotLeft + 16;
  const logEnd = plotRight - 6;
  const logMin = Math.log10(0.08);
  const logMax = Math.log10(36);
  const linMax = 28.5;

  const xScale = (cost) => {
    if (isLog) {
      if (cost <= 0) return logStart;
      const ratio = (Math.log10(cost) - logMin) / (logMax - logMin);
      return logStart + ratio * (logEnd - logStart);
    }
    return plotLeft + (cost / linMax) * (plotRight - plotLeft);
  };

  const svg = svgEl("svg", {
    class: "chart-svg",
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-labelledby": "scatter-svg-title scatter-svg-desc",
  });

  svg.appendChild(svgEl("title", { id: "scatter-svg-title" }, "Composite Benchmark Score vs Task Cost Scatter Plot"));
  svg.appendChild(svgEl("desc", { id: "scatter-svg-desc" },
    "Scatter plot of composite 3-benchmark score versus evaluation suite cost in USD across 27 models, highlighting AWS GovCloud models with crimson-ringed squares, on-prem open-weight non-Chinese models with emerald-ringed diamonds, and the paid Pareto frontier."
  ));

  const gridG = svgEl("g", { "aria-hidden": "true" });

  const yTicks = [20, 30, 40, 50, 60, 70, 80, 90];
  for (const yt of yTicks) {
    const py = yScale(yt);
    gridG.appendChild(svgEl("line", {
      x1: plotLeft, y1: py, x2: plotRight, y2: py,
      stroke: "var(--grid)", "stroke-width": 1, "stroke-dasharray": "3 3",
    }));
    gridG.appendChild(svgEl("text", {
      x: plotLeft - 12, y: py + 4,
      "text-anchor": "end",
      fill: "var(--axis)",
      "font-size": 12,
      "font-weight": 550,
    }, String(yt)));
  }

  const xTicks = isLog
    ? [
        { val: 0.1, label: "$0.10" },
        { val: 0.3, label: "$0.30" },
        { val: 1, label: "$1" },
        { val: 3, label: "$3" },
        { val: 10, label: "$10" },
        { val: 30, label: "$30" },
      ]
    : [
        { val: 0, label: "$0" },
        { val: 5, label: "$5" },
        { val: 10, label: "$10" },
        { val: 15, label: "$15" },
        { val: 20, label: "$20" },
        { val: 25, label: "$25" },
      ];

  for (const xt of xTicks) {
    const px = xScale(xt.val);
    gridG.appendChild(svgEl("line", {
      x1: px, y1: plotTop, x2: px, y2: plotBottom,
      stroke: "var(--grid)", "stroke-width": 1, "stroke-dasharray": "3 3",
    }));
    gridG.appendChild(svgEl("text", {
      x: px, y: plotBottom + 22,
      "text-anchor": "middle",
      fill: "var(--axis)",
      "font-size": 12,
      "font-weight": 550,
    }, xt.label));
  }

  gridG.appendChild(svgEl("line", {
    x1: plotLeft, y1: plotTop, x2: plotLeft, y2: plotBottom,
    stroke: "var(--axis)", "stroke-width": 1.35,
  }));

  gridG.appendChild(svgEl("line", {
    x1: plotLeft, y1: plotBottom, x2: plotRight, y2: plotBottom,
    stroke: "var(--axis)", "stroke-width": 1.35,
  }));

  gridG.appendChild(svgEl("text", {
    x: plotLeft + (plotRight - plotLeft) / 2,
    y: height - 10,
    "text-anchor": "middle",
    fill: "var(--text-muted)",
    "font-size": 12.5,
    "font-weight": 600,
  }, isLog ? "Task Cost (USD, Logarithmic Scale)" : "Task Cost (USD, Linear Scale)"));

  gridG.appendChild(svgEl("text", {
    x: 20,
    y: plotTop + plotH / 2,
    transform: `rotate(-90 20 ${plotTop + plotH / 2})`,
    "text-anchor": "middle",
    fill: "var(--text-muted)",
    "font-size": 12.5,
    "font-weight": 600,
  }, "Composite Benchmark Score (3-Bench Mean)"));

  const bestGov = models.filter((m) => m.tier === "govcloud").sort((a, b) => b.composite - a.composite)[0];
  const bestOpen = models.filter((m) => m.tier === "onprem_open").sort((a, b) => b.composite - a.composite)[0];

  if (bestGov) {
    const gy = yScale(bestGov.composite);
    gridG.appendChild(svgEl("line", {
      x1: plotLeft, y1: gy, x2: plotRight, y2: gy,
      stroke: "var(--tier-govcloud)",
      "stroke-opacity": 0.34,
      "stroke-width": 1.2,
      "stroke-dasharray": "5 5",
    }));
    gridG.appendChild(svgEl("text", {
      x: plotRight - 4,
      y: gy - 6,
      "text-anchor": "end",
      fill: "var(--tier-govcloud)",
      "font-size": 10.8,
      "font-weight": 650,
    }, `Best AWS GovCloud: ${bestGov.composite.toFixed(1)}`));
  }

  if (bestOpen) {
    const oy = yScale(bestOpen.composite);
    gridG.appendChild(svgEl("line", {
      x1: plotLeft, y1: oy, x2: plotRight, y2: oy,
      stroke: "var(--tier-onprem)",
      "stroke-opacity": 0.38,
      "stroke-width": 1.2,
      "stroke-dasharray": "5 5",
    }));
    gridG.appendChild(svgEl("text", {
      x: plotRight - 4,
      y: oy - 6,
      "text-anchor": "end",
      fill: "var(--tier-onprem)",
      "font-size": 10.8,
      "font-weight": 650,
    }, `Best On-Prem Open: ${bestOpen.composite.toFixed(1)}`));
  }

  svg.appendChild(gridG);

  if (state.showFrontier) {
    const frontierModels = state.data.frontier
      .map((slug) => models.find((m) => m.slug === slug))
      .filter(Boolean);

    if (frontierModels.length > 1) {
      const pts = frontierModels.map((m) => `${xScale(m.cost).toFixed(1)},${yScale(m.composite).toFixed(1)}`).join(" ");
      const fg = svgEl("g", { "aria-hidden": "true" });
      fg.appendChild(svgEl("polyline", {
        points: pts,
        fill: "none",
        stroke: "var(--frontier)",
        "stroke-opacity": 0.52,
        "stroke-width": 1.8,
        "stroke-dasharray": "6 4",
        "stroke-linejoin": "round",
      }));

      const last = frontierModels[frontierModels.length - 1];
      const prev = frontierModels[frontierModels.length - 2];
      const midX = (xScale(prev.cost) + xScale(last.cost)) / 2;
      const midY = (yScale(prev.composite) + yScale(last.composite)) / 2;
      fg.appendChild(svgEl("text", {
        x: midX - 8,
        y: midY - 12,
        "text-anchor": "end",
        fill: "var(--text-muted)",
        "font-size": 11,
        "font-weight": 600,
        "font-style": "italic",
      }, "Paid Pareto frontier"));
      svg.appendChild(fg);
    }
  }

  const coords = models.map((m) => ({
    model: m,
    x: xScale(m.cost),
    y: yScale(m.composite),
    r: m.tier === "standard" ? 7.5 : 14.5,
  }));

  // Reserve marker discs and right-side threshold labels before placing model labels.
  const occupied = coords.map((c) => ({
    x: c.x - c.r,
    y: c.y - c.r,
    w: c.r * 2,
    h: c.r * 2,
  }));
  if (bestGov) {
    occupied.push({ x: plotRight - 165, y: yScale(bestGov.composite) - 18, w: 165, h: 16 });
  }
  if (bestOpen) {
    occupied.push({ x: plotRight - 165, y: yScale(bestOpen.composite) - 18, w: 165, h: 16 });
  }

  const labelOrder = [...coords].sort((a, b) => {
    const prio = (m) => {
      if (m.tier === "govcloud") return 0;
      if (m.tier === "onprem_open") return 1;
      if (frontierSet.has(m.slug)) return 2;
      if (m.rank <= 6) return 3;
      return 4;
    };
    const pa = prio(a.model);
    const pb = prio(b.model);
    if (pa !== pb) return pa - pb;
    return a.model.rank - b.model.rank;
  });

  const offsets = [
    { dx: 14, dy: -10, anchor: "start" },
    { dx: 14, dy: 14, anchor: "start" },
    { dx: -14, dy: -10, anchor: "end" },
    { dx: -14, dy: 14, anchor: "end" },
    { dx: 0, dy: -18, anchor: "middle" },
    { dx: 0, dy: 22, anchor: "middle" },
    { dx: 20, dy: -22, anchor: "start" },
    { dx: -20, dy: -22, anchor: "end" },
    { dx: 20, dy: 24, anchor: "start" },
    { dx: -20, dy: 24, anchor: "end" },
    { dx: 30, dy: 0, anchor: "start" },
    { dx: -30, dy: 0, anchor: "end" },
    { dx: 0, dy: -32, anchor: "middle" },
    { dx: 0, dy: 34, anchor: "middle" },
    { dx: 36, dy: -26, anchor: "start" },
    { dx: -36, dy: -26, anchor: "end" },
    { dx: 36, dy: 28, anchor: "start" },
    { dx: -36, dy: 28, anchor: "end" },
    { dx: 50, dy: -36, anchor: "start" },
    { dx: -50, dy: 36, anchor: "end" },
    { dx: 64, dy: -16, anchor: "start" },
    { dx: 64, dy: 20, anchor: "start" },
    { dx: 64, dy: 44, anchor: "start" },
    { dx: 82, dy: -42, anchor: "start" },
    { dx: 82, dy: 52, anchor: "start" },
    { dx: -64, dy: -38, anchor: "end" },
    { dx: 110, dy: 12, anchor: "start" },
    { dx: 110, dy: -28, anchor: "start" },
    { dx: 110, dy: 48, anchor: "start" },
  ];

  const placedLabels = new Map();

  for (const item of labelOrder) {
    const m = item.model;
    const isSpecial = m.tier !== "standard";
    const text = isSpecial ? `${m.name} · ${TIER_INFO[m.tier].scatterTag}` : m.name;
    const boxW = estimateLabelWidth(text, isSpecial);
    const boxH = isSpecial ? 21 : 15;
    const extraPad = isSpecial ? 7 : 1;

    let chosen = null;
    for (const off of offsets) {
      const dx = off.dx + (off.dx > 0 ? extraPad : off.dx < 0 ? -extraPad : 0);
      const dy = off.dy + (off.dy > 0 ? extraPad : off.dy < 0 ? -extraPad : 0);
      const tx = item.x + dx;
      const ty = item.y + dy;
      let bx = tx;
      if (off.anchor === "end") bx = tx - boxW;
      if (off.anchor === "middle") bx = tx - boxW / 2;
      const by = ty - boxH * 0.72;

      if (bx < plotLeft + 2 || bx + boxW > plotRight - 2 || by < plotTop + 2 || by + boxH > plotBottom - 2) {
        continue;
      }

      const candidate = { x: bx, y: by, w: boxW, h: boxH };
      const collides = occupied.some((occ) => rectsOverlap(candidate, occ, 3));
      if (!collides) {
        chosen = {
          text,
          tx,
          ty,
          anchor: off.anchor,
          box: candidate,
          dist: Math.hypot(dx, dy),
          isSpecial,
        };
        break;
      }
    }

    if (!chosen && isSpecial) {
      // Search progressively further right and vertically so Honeywell callouts never overlap.
      for (const shiftX of [130, 175, 220, 265]) {
        for (const shiftY of [0, -24, 24, -48, 48, -72, 72]) {
          const tx = Math.min(plotRight - boxW - 4, Math.max(plotLeft + 4, item.x + shiftX));
          const ty = Math.min(plotBottom - 8, Math.max(plotTop + 18, item.y + shiftY));
          const candidate = { x: tx, y: ty - boxH * 0.72, w: boxW, h: boxH };
          if (!occupied.some((occ) => rectsOverlap(candidate, occ, 3))) {
            chosen = { text, tx, ty, anchor: "start", box: candidate, dist: Math.hypot(shiftX, shiftY), isSpecial: true };
            break;
          }
        }
        if (chosen) break;
      }
    }

    if (chosen) {
      occupied.push(chosen.box);
      placedLabels.set(m.slug, chosen);
    }
  }

  // Draw standard points first so Honeywell highlighted markers sit cleanly on top.
  const drawOrder = [...coords].sort((a, b) => {
    const rank = (m) => (m.tier === "standard" ? 0 : m.tier === "onprem_open" ? 1 : 2);
    return rank(a.model) - rank(b.model);
  });

  const pointsG = svgEl("g");
  const labelsG = svgEl("g");

  for (const item of drawOrder) {
    const m = item.model;
    const isGov = m.tier === "govcloud";
    const isOpen = m.tier === "onprem_open";
    const highlighted = isModelHighlighted(m);
    const dimClass = highlighted ? "" : " is-dimmed";
    const hoverClass = state.hoveredSlug === m.slug ? " is-hovered" : "";

    const g = svgEl("g", {
      class: `scatter-item${dimClass}${hoverClass}`,
      "data-slug": m.slug,
      tabindex: "0",
      role: "button",
      "aria-label": `${m.name} (${m.lab}), composite score ${m.composite.toFixed(1)}, task cost ${formatCost(m.cost)}${m.tier !== "standard" ? ", " + TIER_INFO[m.tier].tableText : ""}`,
    });

    g.appendChild(svgEl("circle", {
      class: "point-focus-ring",
      cx: item.x,
      cy: item.y,
      r: isGov || isOpen ? 18 : 11,
      fill: "none",
      stroke: "var(--focus-ring)",
      "stroke-width": 2.4,
      "stroke-opacity": 0,
    }));

    if (isGov) {
      g.appendChild(svgEl("circle", {
        cx: item.x,
        cy: item.y,
        r: 13.5,
        fill: "var(--tier-govcloud-bg)",
        stroke: "var(--tier-govcloud)",
        "stroke-width": 2.2,
      }));
      g.appendChild(svgEl("rect", {
        class: "point-main",
        x: item.x - 5.5,
        y: item.y - 5.5,
        width: 11,
        height: 11,
        rx: 2.5,
        fill: labColor(m.labKey),
        stroke: "var(--surface)",
        "stroke-width": 1.6,
      }));
    } else if (isOpen) {
      g.appendChild(svgEl("circle", {
        cx: item.x,
        cy: item.y,
        r: 13.5,
        fill: "var(--tier-onprem-bg)",
        stroke: "var(--tier-onprem)",
        "stroke-width": 2.2,
      }));
      g.appendChild(svgEl("polygon", {
        class: "point-main",
        points: `${item.x},${item.y - 7.5} ${item.x + 7.5},${item.y} ${item.x},${item.y + 7.5} ${item.x - 7.5},${item.y}`,
        fill: labColor(m.labKey),
        stroke: "var(--surface)",
        "stroke-width": 1.6,
      }));
    } else {
      g.appendChild(svgEl("circle", {
        class: "point-main",
        cx: item.x,
        cy: item.y,
        r: 6.5,
        fill: labColor(m.labKey),
        stroke: "var(--surface)",
        "stroke-width": 1.8,
      }));
    }

    bindInteractiveEvents(g, m);
    pointsG.appendChild(g);

    const lbl = placedLabels.get(m.slug);
    if (lbl) {
      const lg = svgEl("g", {
        class: `scatter-item${dimClass}${hoverClass}`,
        "data-slug": m.slug,
        "pointer-events": "none",
      });

      if (lbl.dist > 18 || lbl.isSpecial) {
        const edgeX = lbl.anchor === "start" ? lbl.box.x : lbl.anchor === "end" ? lbl.box.x + lbl.box.w : lbl.box.x + lbl.box.w / 2;
        const edgeY = lbl.box.y + lbl.box.h / 2;
        lg.appendChild(svgEl("line", {
          x1: item.x,
          y1: item.y,
          x2: edgeX,
          y2: edgeY,
          stroke: isGov ? "var(--tier-govcloud)" : isOpen ? "var(--tier-onprem)" : "var(--axis)",
          "stroke-opacity": lbl.isSpecial ? 0.65 : 0.4,
          "stroke-width": lbl.isSpecial ? 1.3 : 1,
        }));
      }

      if (lbl.isSpecial) {
        const accent = isGov ? "var(--tier-govcloud)" : "var(--tier-onprem)";
        const bg = isGov ? "var(--tier-govcloud-bg)" : "var(--tier-onprem-bg)";
        lg.appendChild(svgEl("rect", {
          x: lbl.box.x,
          y: lbl.box.y,
          width: lbl.box.w,
          height: lbl.box.h,
          rx: 10.5,
          fill: bg,
          stroke: accent,
          "stroke-width": 1.3,
        }));
        lg.appendChild(svgEl("text", {
          x: lbl.box.x + lbl.box.w / 2,
          y: lbl.box.y + lbl.box.h * 0.7,
          "text-anchor": "middle",
          fill: accent,
          "font-size": 11.2,
          "font-weight": 700,
        }, lbl.text));
      } else {
        lg.appendChild(svgEl("text", {
          x: lbl.tx,
          y: lbl.ty,
          "text-anchor": lbl.anchor,
          fill: "var(--text)",
          "font-size": 11.2,
          "font-weight": frontierSet.has(m.slug) ? 650 : 500,
        }, lbl.text));
      }

      labelsG.appendChild(lg);
    }
  }

  svg.appendChild(pointsG);
  svg.appendChild(labelsG);
  container.replaceChildren(svg);
}

function renderDataTable() {
  const tbody = document.getElementById("benchmark-table-body");
  const frag = document.createDocumentFragment();

  for (const m of state.data.models) {
    const tr = document.createElement("tr");
    if (m.tier === "govcloud") tr.className = "row-govcloud";
    if (m.tier === "onprem_open") tr.className = "row-onprem";

    const cells = [
      { tag: "td", text: String(m.rank), label: "Rank" },
      { tag: "th", text: m.name, scope: "row", label: "Model" },
      { tag: "td", text: m.lab, label: "Lab" },
      { tag: "td", text: TIER_INFO[m.tier].tableText, label: "Honeywell Status" },
      { tag: "td", text: m.adev.toFixed(1), cls: "num", label: "ADEV" },
      { tag: "td", text: m.tbs.toFixed(1), cls: "num", label: "TB Science" },
      { tag: "td", text: m.trivia.toFixed(1), cls: "num", label: "Trivia" },
      { tag: "td", text: m.composite.toFixed(2), cls: "num", label: "Composite" },
      { tag: "td", text: formatCost(m.cost), cls: "num", label: "Task Cost" },
    ];

    for (const c of cells) {
      const el = document.createElement(c.tag);
      el.textContent = c.text;
      if (c.scope) el.scope = c.scope;
      if (c.cls) el.className = c.cls;
      if (c.tag === "td") el.dataset.label = c.label;
      tr.appendChild(el);
    }
    frag.appendChild(tr);
  }

  tbody.replaceChildren(frag);
}

function renderLabLegend() {
  const host = document.getElementById("lab-filters");
  const counts = new Map();
  for (const m of state.data.models) {
    counts.set(m.labKey, (counts.get(m.labKey) || 0) + 1);
  }

  const frag = document.createDocumentFragment();
  for (const lab of LAB_META) {
    const count = counts.get(lab.key) || 0;
    if (!count) continue;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "lab-pill";
    btn.dataset.lab = lab.key;
    btn.setAttribute("aria-pressed", String(state.labFilter === lab.key));

    const swatch = document.createElement("span");
    swatch.className = "lab-swatch";
    swatch.style.background = `var(${lab.cssVar})`;
    swatch.setAttribute("aria-hidden", "true");

    const label = document.createElement("span");
    label.textContent = lab.label;

    const num = document.createElement("span");
    num.className = "pill-count";
    num.textContent = String(count);

    btn.append(swatch, label, num);
    btn.addEventListener("click", () => {
      state.labFilter = state.labFilter === lab.key ? null : lab.key;
      updateFilterButtons();
      renderCharts();
    });
    frag.appendChild(btn);
  }
  host.replaceChildren(frag);
}

function updateFilterButtons() {
  document.querySelectorAll("#tier-filters .filter-pill").forEach((btn) => {
    btn.setAttribute("aria-pressed", String(btn.dataset.tier === state.tierFilter));
  });
  document.querySelectorAll("#lab-filters .lab-pill").forEach((btn) => {
    btn.setAttribute("aria-pressed", String(btn.dataset.lab === state.labFilter));
  });
}

function renderCharts() {
  renderBarChart();
  renderScatterChart();
}

const tooltipEl = document.getElementById("chart-tooltip");
const tooltipContent = document.getElementById("tooltip-content");
const supportsPopover = Object.hasOwn(HTMLElement.prototype, "popover");

function positionTooltip(clientX, clientY) {
  const pad = 14;
  const rect = tooltipEl.getBoundingClientRect();
  let left = clientX + 16;
  let top = clientY - rect.height - 14;

  if (left + rect.width > window.innerWidth - pad) {
    left = clientX - rect.width - 16;
  }
  if (left < pad) {
    left = pad;
  }
  if (top < pad) {
    top = clientY + 18;
  }
  if (top + rect.height > window.innerHeight - pad) {
    top = Math.max(pad, window.innerHeight - rect.height - pad);
  }

  tooltipEl.style.left = `${Math.round(left)}px`;
  tooltipEl.style.top = `${Math.round(top)}px`;
}

function showTooltip(model, anchorEvent) {
  const tierBlock = model.tier !== "standard"
    ? `<div class="tt-tier ${model.tier}">${TIER_INFO[model.tier].tooltipText}</div>`
    : "";

  tooltipContent.innerHTML = `
    <div class="tt-head">
      <span class="tt-title">
        <span class="lab-swatch" style="background: ${labColor(model.labKey)}" aria-hidden="true"></span>
        ${model.name}
      </span>
      <span class="tt-rank">#${model.rank} · ${model.lab}</span>
    </div>
    ${tierBlock}
    <div class="tt-grid">
      <div class="tt-metric"><span class="tt-metric-label">Composite</span><span class="tt-metric-val">${model.composite.toFixed(2)}</span></div>
      <div class="tt-metric"><span class="tt-metric-label">Task Cost</span><span class="tt-metric-val">${formatCost(model.cost)}</span></div>
      <div class="tt-metric"><span class="tt-metric-label">ADEV Bench</span><span class="tt-metric-val">${model.adev.toFixed(1)}</span></div>
      <div class="tt-metric"><span class="tt-metric-label">TB Science</span><span class="tt-metric-val">${model.tbs.toFixed(1)}</span></div>
      <div class="tt-metric"><span class="tt-metric-label">Trivia Bench</span><span class="tt-metric-val">${model.trivia.toFixed(1)}</span></div>
      <div class="tt-metric"><span class="tt-metric-label">Eval Date</span><span class="tt-metric-val">${model.date}</span></div>
    </div>
    <div class="tt-foot">
      <span>${model.provider}</span>
      <span>${model.slug}</span>
    </div>
  `;

  tooltipEl.hidden = false;
  if (supportsPopover) {
    try {
      if (!tooltipEl.matches(":popover-open")) {
        tooltipEl.showPopover();
      }
    } catch {
      // Ignore if popover is already open or transitioning.
    }
  }

  if (anchorEvent && typeof anchorEvent.clientX === "number") {
    positionTooltip(anchorEvent.clientX, anchorEvent.clientY);
  } else if (anchorEvent && anchorEvent.currentTarget) {
    const box = anchorEvent.currentTarget.getBoundingClientRect();
    positionTooltip(box.left + box.width / 2, box.top);
  }
}

function hideTooltip() {
  if (supportsPopover) {
    try {
      if (tooltipEl.matches(":popover-open")) {
        tooltipEl.hidePopover();
      }
    } catch {
      // Ignore.
    }
  }
  tooltipEl.hidden = true;
}

function setHoveredModel(slug) {
  state.hoveredSlug = slug;
  document.querySelectorAll("[data-slug]").forEach((el) => {
    el.classList.toggle("is-hovered", Boolean(slug) && el.dataset.slug === slug);
  });
}

function bindInteractiveEvents(el, model) {
  el.addEventListener("pointerenter", (e) => {
    setHoveredModel(model.slug);
    showTooltip(model, e);
  });
  el.addEventListener("pointermove", (e) => {
    if (state.hoveredSlug === model.slug) {
      positionTooltip(e.clientX, e.clientY);
    }
  });
  el.addEventListener("pointerleave", () => {
    setHoveredModel(null);
    hideTooltip();
  });
  el.addEventListener("focus", (e) => {
    setHoveredModel(model.slug);
    showTooltip(model, e);
  });
  el.addEventListener("blur", () => {
    setHoveredModel(null);
    hideTooltip();
  });
}

function initThemeToggle() {
  const btn = document.getElementById("theme-toggle");
  const meta = document.querySelector('meta[name="color-scheme"]');
  const mq = window.matchMedia("(prefers-color-scheme: dark)");

  function getSystemScheme() {
    return mq.matches ? "dark" : "light";
  }

  function getRenderedScheme() {
    if (meta.content === "dark" || meta.content === "light") {
      return meta.content;
    }
    return getSystemScheme();
  }

  btn.addEventListener("click", () => {
    const target = getRenderedScheme() === "dark" ? "light" : "dark";
    if (target === getSystemScheme()) {
      try {
        localStorage.removeItem("color-scheme");
      } catch {
        // Storage may be blocked; apply in memory only.
      }
      meta.content = "light dark";
    } else {
      try {
        localStorage.setItem("color-scheme", target);
      } catch {
        // Storage may be blocked; apply in memory only.
      }
      meta.content = target;
    }
  });

  window.addEventListener("storage", (event) => {
    if (event.key === "color-scheme") {
      meta.content = event.newValue === "light" || event.newValue === "dark" ? event.newValue : "light dark";
    }
  });
}

function initControls() {
  document.querySelectorAll("#tier-filters .filter-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.tierFilter = btn.dataset.tier;
      updateFilterButtons();
      renderCharts();
    });
  });

  document.querySelectorAll(".seg-btn[data-scale]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.scale = btn.dataset.scale;
      document.querySelectorAll(".seg-btn[data-scale]").forEach((b) => {
        b.setAttribute("aria-pressed", String(b.dataset.scale === state.scale));
      });
      renderScatterChart();
    });
  });

  const frontierBtn = document.getElementById("frontier-toggle");
  frontierBtn.addEventListener("click", () => {
    state.showFrontier = !state.showFrontier;
    frontierBtn.setAttribute("aria-pressed", String(state.showFrontier));
    renderScatterChart();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      setHoveredModel(null);
      hideTooltip();
    }
  });
}

async function main() {
  initThemeToggle();
  initControls();

  const res = await fetch("/api/benchmarks.json");
  if (!res.ok) {
    throw new Error(`Failed to load benchmark data (${res.status})`);
  }
  state.data = await res.json();

  const [startDate, endDate] = state.data.meta.dateRange || [];
  if (startDate && endDate) {
    document.getElementById("suite-version").textContent = `${state.data.meta.version} · ${startDate} to ${endDate}`;
  }

  renderLabLegend();
  renderCharts();
  renderDataTable();
}

main().catch((err) => {
  const stage = document.getElementById("bar-chart-stage");
  stage.textContent = `Could not load benchmark data: ${err.message}`;
});
