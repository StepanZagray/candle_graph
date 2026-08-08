(function () {
  "use strict";

  const P = JSON.parse(document.getElementById("cg-payload").textContent);
  const V = P.views || {};
  const SUM = P.summary || {};

  const VIEW_META = [
    { id: "evidence", label: "Evidence", graph: false },
    { id: "trace", label: "Trace", graph: true, layout: "layered", direction: "LR" },
    { id: "span_costs", label: "Span costs", graph: false },
    { id: "memory", label: "Memory", graph: false },
    { id: "gpu", label: "GPU", graph: false },
  ];

  let currentView = P.default_view || "evidence";
  let heatMode = "time";
  let selectedId = null;
  let hoveredId = null;
  let graphState = null;
  let graphView = { x: 0, y: 0, k: 1 };
  const spanOpen = new Set();

  const root = document.documentElement;
  const pref = matchMedia("(prefers-color-scheme:dark)").matches ? "dark" : "light";
  root.setAttribute("data-theme", localStorage.getItem("cg-theme") || pref);

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch])
    );
  }
  function idStr(v) { return v == null ? "" : String(v); }
  function fmtMs(ms) {
    if (ms == null || !Number.isFinite(ms)) return "—";
    return Number(ms).toFixed(2) + " ms";
  }
  function fmtShape(s) {
    if (s == null) return "—";
    return Array.isArray(s) ? s.join(" × ") : String(s);
  }
  function fmtBytes(n) {
    if (n == null || !Number.isFinite(n) || n <= 0) return "—";
    var b = Number(n);
    if (b >= 1073741824) return (b / 1073741824).toFixed(2) + " GiB";
    if (b >= 1048576) return (b / 1048576).toFixed(2) + " MiB";
    if (b >= 1024) return (b / 1024).toFixed(1) + " KiB";
    return b + " B";
  }
  function fmtNsMs(ns) {
    if (ns == null || !Number.isFinite(ns)) return "—";
    return (Number(ns) / 1e6).toFixed(2) + " ms";
  }
  function isScalar(value) {
    return value == null || ["string", "number", "boolean"].includes(typeof value);
  }
  function humanize(value) {
    return String(value || "")
      .replace(/_/g, " ")
      .replace(/\b\w/g, function (ch) { return ch.toUpperCase(); });
  }
  function fmtValue(value) {
    if (value == null || value === "") return "—";
    if (typeof value === "boolean") return value ? "Yes" : "No";
    if (Array.isArray(value)) {
      return value.map(function (item) { return isScalar(item) ? fmtValue(item) : JSON.stringify(item); }).join(", ") || "—";
    }
    if (typeof value === "object") {
      return Object.keys(value).map(function (key) {
        return humanize(key) + ": " + fmtValue(value[key]);
      }).join(" · ") || "—";
    }
    return String(value);
  }
  function safeStatus(value) {
    return String(value || "unknown").toLowerCase().replace(/[^a-z0-9_-]/g, "-");
  }
  function statusLabel(value) {
    var status = safeStatus(value);
    var marks = {
      valid: "✓", trusted: "✓", available: "✓", captured: "✓", complete: "✓",
      warning: "!", partial: "!", untrusted: "!", invalid: "×", unavailable: "—",
      absent: "—", missing: "—", unknown: "?",
    };
    return '<span class="status-badge status-' + esc(status) + '"><span aria-hidden="true">' +
      esc(marks[status] || "•") + '</span> ' + esc(humanize(status)) + "</span>";
  }

  function heatColor(n) {
    if (heatMode === "memory") {
      var peak = (SUM.memory && SUM.memory.peak_bytes) || 1;
      var ratio = Math.max(0, Math.min(1, (n.peak_bytes || n.bytes || 0) / peak));
      var hue = (1 - ratio) * 220;
      return "hsl(" + hue.toFixed(0) + ", 68%, 42%)";
    }
    return heatStroke(n);
  }

  function heatStroke(n) {
    var ratio = n.self_ratio;
    if (ratio == null && n.total_time_ns > 0) {
      ratio = (n.self_time_ns || 0) / n.total_time_ns;
    }
    ratio = Math.max(0, Math.min(1, ratio || 0));
    var hue = (1 - ratio) * 220;
    return "hsl(" + hue.toFixed(0) + ", 68%, 42%)";
  }

  function initResizers() {
    var sidebar = document.querySelector(".pane-sidebar");
    var inspector = document.querySelector(".pane-inspector");
    document.querySelectorAll(".resize-handle").forEach(function (handle) {
      handle.addEventListener("pointerdown", function (e) {
        e.preventDefault();
        handle.classList.add("active");
        var side = handle.dataset.side;
        var startX = e.clientX;
        var startW = side === "left" ? sidebar.offsetWidth : inspector.offsetWidth;
        function move(ev) {
          var dx = ev.clientX - startX;
          if (side === "left") {
            var w = Math.max(200, Math.min(window.innerWidth * 0.45, startW + dx));
            root.style.setProperty("--sidebar-w", w + "px");
          } else {
            var w = Math.max(200, Math.min(window.innerWidth * 0.45, startW - dx));
            root.style.setProperty("--inspector-w", w + "px");
          }
        }
        function up() {
          handle.classList.remove("active");
          window.removeEventListener("pointermove", move);
          window.removeEventListener("pointerup", up);
        }
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
      });
    });
  }

  function renderCoverage() {
    var el = document.querySelector("[data-coverage]");
    if (!el) return;
    el.textContent = [
      SUM.entrypoint,
      SUM.total_ms != null ? fmtMs(SUM.total_ms) + " total" : null,
      SUM.memory && SUM.memory.peak_bytes ? fmtBytes(SUM.memory.peak_bytes) + " peak" : null,
      SUM.phase || null,
    ].filter(Boolean).join(" · ") || "Trace";
  }

  function renderPeakBreakdown() {
    var panel = document.getElementById("peak-breakdown");
    if (!panel) return;
    var rows = (SUM.peak_breakdown || P.views.memory && P.views.memory.peak_breakdown) || [];
    if (!rows.length) {
      panel.innerHTML = "<p class=\"section-empty\">No peak allocations recorded.</p>";
      return;
    }
    panel.innerHTML =
      '<table><thead><tr><th>Tensor</th><th>Op</th><th>Size</th><th>Shape</th></tr></thead><tbody>' +
      rows.map(function (r) {
        return "<tr><td>" + esc(r.tensor_id || "") + "</td><td>" + esc(r.op_name || "—") +
          "</td><td>" + esc(fmtBytes(r.bytes)) + "</td><td>" + esc(fmtShape(r.shape)) + "</td></tr>";
      }).join("") +
      "</tbody></table>";
  }

  function setInspector(o) {
    var empty = !o;
    var insp = document.getElementById("inspector");
    if (insp) insp.classList.toggle("is-empty", empty);
    o = o || {};
    var fields = {
      label: empty ? "Nothing selected" : (o.label || o.name || "—"),
      kind: o.kind || "—",
      self_time: fmtMs(o.self_time_ms != null ? o.self_time_ms : (o.self_ms != null ? o.self_ms : null)),
      total_time: fmtMs(o.total_time_ms != null ? o.total_time_ms : (o.total_ms != null ? o.total_ms : null)),
      shape: fmtShape(o.shape),
      dtype: o.dtype || "—",
      storage: fmtBytes(o.storage_bytes),
      peak_bytes: fmtBytes(o.peak_bytes),
      bytes: fmtBytes(o.bytes),
    };
    Object.keys(fields).forEach(function (k) {
      var el = document.querySelector('[data-field="' + k + '"]');
      if (el) el.textContent = fields[k];
    });
  }

  function initTabs() {
    var tabs = document.querySelector("[data-view-tabs]");
    if (!tabs) return;
    tabs.innerHTML = VIEW_META.map(function (m) {
      return '<button type="button" role="tab" class="tab" id="view-tab-' + esc(m.id) +
        '" data-view="' + esc(m.id) + '" aria-controls="view-panel-' + esc(m.id) +
        '" aria-selected="false" tabindex="-1">' + esc(m.label) + "</button>";
    }).join("");
    tabs.onclick = function (e) {
      var btn = e.target.closest("[data-view]");
      if (!btn) return;
      selectView(btn.dataset.view, true);
    };
    tabs.onkeydown = function (e) {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
      var buttons = Array.from(tabs.querySelectorAll('[role="tab"]'));
      var active = buttons.indexOf(document.activeElement);
      if (active < 0) return;
      e.preventDefault();
      var next = active;
      if (e.key === "Home") next = 0;
      else if (e.key === "End") next = buttons.length - 1;
      else if (e.key === "ArrowLeft") next = (active - 1 + buttons.length) % buttons.length;
      else next = (active + 1) % buttons.length;
      selectView(buttons[next].dataset.view, true);
      buttons[next].focus();
    };
  }

  function selectView(id, push) {
    var meta = VIEW_META.find(function (m) { return m.id === id; }) || VIEW_META[0];
    currentView = meta.id;
    document.querySelectorAll("[data-view-tabs] [data-view]").forEach(function (b) {
      var selected = b.dataset.view === currentView;
      b.setAttribute("aria-selected", selected ? "true" : "false");
      b.setAttribute("tabindex", selected ? "0" : "-1");
    });
    document.querySelectorAll("[data-view-panel]").forEach(function (panel) {
      panel.hidden = panel.dataset.viewPanel !== currentView;
    });
    document.querySelectorAll("[data-trace-only]").forEach(function (control) {
      control.hidden = currentView !== "trace";
    });
    var traceNavigation = document.getElementById("trace-navigation");
    if (traceNavigation) traceNavigation.hidden = currentView !== "trace";
    var inspector = document.querySelector(".pane-inspector");
    var inspectorVisible = currentView === "trace" || currentView === "span_costs";
    if (inspector) inspector.hidden = !inspectorVisible;
    var rightHandle = document.querySelector('.resize-handle[data-side="right"]');
    if (rightHandle) rightHandle.hidden = !inspectorVisible;
    var title = document.querySelector("[data-canvas-title]");
    if (title) title.textContent = meta.label;
    var stats = document.querySelector("[data-graph-stats]");
    if (stats && currentView !== "trace") stats.textContent = "";
    refreshView();
    if (push) setInspector(null);
  }

  function refreshView() {
    var meta = VIEW_META.find(function (m) { return m.id === currentView; }) || VIEW_META[0];
    if (meta.graph) {
      drawGraph(V.trace || { nodes: [], edges: [] }, meta);
      return;
    }
    graphState = null;
    if (meta.id === "evidence") renderEvidenceView(V.evidence || {});
    else if (meta.id === "span_costs") renderSpanCosts(V.span_costs || { items: [] });
    else if (meta.id === "memory") renderMemoryView(V.memory || { timeline: [], peak_breakdown: [], summary: {} });
    else if (meta.id === "gpu") renderGpuView(V.gpu || {});
  }

  function panelFor(id) {
    return document.querySelector('[data-view-panel="' + id + '"]');
  }

  function findSpanRow(tree, id) {
    return Array.from(tree.querySelectorAll("[data-span-id]")).find(function (row) {
      return row.dataset.spanId === id;
    }) || null;
  }

  function formatCell(key, value) {
    var lower = String(key || "").toLowerCase();
    if (typeof value === "number" && lower.includes("bytes")) return fmtBytes(value);
    if (typeof value === "number" && lower.endsWith("_ns")) return fmtNsMs(value);
    if (typeof value === "number" && lower.endsWith("_ms")) return fmtMs(value);
    if (typeof value === "number" && lower.includes("percent")) return value.toFixed(2) + "%";
    return fmtValue(value);
  }

  function renderKeyValues(record, omitted) {
    var skip = new Set(omitted || []);
    var entries = Object.keys(record || {}).filter(function (key) {
      return !skip.has(key) && !Array.isArray(record[key]);
    });
    if (!entries.length) return '<p class="section-empty">No details recorded.</p>';
    return '<dl class="key-values">' + entries.map(function (key) {
      return '<div><dt>' + esc(humanize(key)) + '</dt><dd>' + esc(formatCell(key, record[key])) + '</dd></div>';
    }).join("") + "</dl>";
  }

  function renderNoticeList(items, kind, emptyText) {
    items = Array.isArray(items) ? items : [];
    if (!items.length) return '<p class="section-empty">' + esc(emptyText) + "</p>";
    return '<ul class="notice-list" role="list">' + items.map(function (item) {
      var record = isScalar(item) ? { message: item } : (item || {});
      var severity = record.severity || record.status || kind;
      var title = record.title || record.code || record.name || humanize(severity);
      var message = record.message || record.detail || record.description || "";
      return '<li class="notice notice-' + esc(safeStatus(severity)) + '">' +
        '<div class="notice-title">' + statusLabel(severity) + '<strong>' + esc(title) + '</strong></div>' +
        (message ? '<p>' + esc(message) + "</p>" : "") + "</li>";
    }).join("") + "</ul>";
  }

  function renderDataTable(rows, caption, limit) {
    rows = Array.isArray(rows) ? rows : [];
    if (!rows.length) return '<p class="section-empty">No ' + esc(caption.toLowerCase()) + " recorded.</p>";
    var capped = rows.slice(0, limit || 100);
    var keys = [];
    capped.forEach(function (row) {
      var record = isScalar(row) ? { value: row } : (row || {});
      Object.keys(record).forEach(function (key) {
        if (!keys.includes(key)) keys.push(key);
      });
    });
    var html = '<div class="table-wrap"><table class="data-table"><caption class="sr">' + esc(caption) +
      '</caption><thead><tr>' + keys.map(function (key) { return "<th scope=\"col\">" + esc(humanize(key)) + "</th>"; }).join("") +
      "</tr></thead><tbody>";
    html += capped.map(function (row) {
      var record = isScalar(row) ? { value: row } : (row || {});
      return "<tr>" + keys.map(function (key) {
        return "<td>" + esc(formatCell(key, record[key])) + "</td>";
      }).join("") + "</tr>";
    }).join("");
    html += "</tbody></table></div>";
    if (rows.length > capped.length) {
      html += '<p class="table-note">Showing ' + capped.length + " of " + rows.length + " rows.</p>";
    }
    return html;
  }

  function renderComparison(comparison) {
    if (!comparison || (typeof comparison === "object" && !Object.keys(comparison).length)) {
      return '<p class="section-empty">No baseline comparison was requested for this run.</p>';
    }
    var rows = comparison.items || comparison.spans || comparison.span_deltas || comparison.deltas || [];
    var omitted = ["items", "spans", "span_deltas", "deltas", "warnings"];
    var html = renderKeyValues(comparison, omitted);
    if (comparison.warnings) html += renderNoticeList(comparison.warnings, "warning", "No comparison warnings.");
    if (rows.length) html += renderDataTable(rows, "Baseline comparison", 100);
    return html;
  }

  function renderEvidenceView(data) {
    var panel = panelFor("evidence");
    if (!panel) return;
    var provenance = data.provenance || {};
    var health = data.health || {};
    var healthStatus = health.status || health.state ||
      (health.valid === false ? "invalid" : health.trusted === false ? "untrusted" :
        (health.valid || health.trusted) ? "trusted" : "unknown");
    var issues = health.issues || health.problems || [];
    var findings = data.findings || [];
    var gaps = data.gaps || [];
    panel.innerHTML =
      '<div class="evidence-header"><div><p class="eyebrow">Representative profile run</p>' +
      '<h1>Profile evidence</h1><p>Start here to decide what this run can support before inspecting individual spans.</p></div>' +
      '<div class="health-summary" role="status" aria-label="Trace health: ' + esc(humanize(healthStatus)) + '">' +
      statusLabel(healthStatus) + '<span class="health-copy">' + esc(health.summary || health.message || "Trace health is recorded with this packet.") +
      "</span></div></div>" +
      '<div class="evidence-grid"><section class="evidence-card evidence-card-wide"><h2>Provenance</h2>' +
      renderKeyValues(provenance) + '</section><section class="evidence-card"><h2>Trace health</h2>' +
      renderKeyValues(health, ["status", "state", "summary", "message", "issues", "problems"]) +
      renderNoticeList(issues, "warning", "No health issues were recorded.") + "</section>" +
      '<section class="evidence-card"><h2>Trusted findings</h2>' +
      renderNoticeList(findings, "information", "No trusted findings were derived from this profile.") + "</section>" +
      '<section class="evidence-card"><h2>Evidence gaps</h2>' +
      renderNoticeList(gaps, "missing", "No evidence gaps were reported.") + "</section>" +
      '<section class="evidence-card evidence-card-wide"><h2>Structured facts</h2>' +
      renderDataTable(data.facts, "Structured facts", 100) + "</section>" +
      '<section class="evidence-card evidence-card-wide"><h2>Tensor checkpoints</h2>' +
      renderDataTable(data.tensors, "Tensor checkpoints", 100) + "</section>" +
      '<section class="evidence-card evidence-card-wide"><h2>Gradient evidence</h2>' +
      renderDataTable(data.gradients, "Gradient evidence", 250) + "</section>" +
      '<section class="evidence-card evidence-card-wide"><h2>Baseline comparison</h2>' +
      renderComparison(data.comparison) + "</section></div>";
  }

  function renderMemoryView(data) {
    var panel = panelFor("memory");
    if (!panel) return;
    var timeline = data.timeline || [];
    var summary = data.summary || {};
    var html = "";
    html += '<div class="view-heading"><div><p class="eyebrow">Allocation evidence</p><h1>Memory</h1></div></div>';
    html += '<p class="view-summary">';
    html += "Peak: <strong>" + esc(fmtBytes(summary.peak_bytes)) + "</strong>";
    if (summary.peak_timestamp_ns) html += " @ " + esc(fmtNsMs(summary.peak_timestamp_ns));
    html += " · allocs " + esc(String(summary.alloc_count || 0));
    html += " · frees " + esc(String(summary.free_count || 0));
    if (summary.autograd_retained_bytes) {
      html += " · autograd retained " + esc(fmtBytes(summary.autograd_retained_bytes));
    }
    html += "</p>";
    var cats = summary.peak_by_category || {};
    var catKeys = Object.keys(cats);
    if (catKeys.length) {
      html += "<p style=\"font-size:11px;margin:0 0 8px;color:var(--muted)\">Peak by category: ";
      html += catKeys.map(function (k) { return esc(k) + " " + esc(fmtBytes(cats[k])); }).join(" · ");
      html += "</p>";
    }
    if (!timeline.length) {
      html += '<div class="content-empty" role="status"><strong>No memory timeline</strong><p>No allocation events were captured for this run.</p></div>';
      panel.innerHTML = html;
      return;
    }
    var maxTs = Math.max.apply(null, timeline.map(function (p) { return p.timestamp_ns || 0; }).concat([1]));
    var maxLive = Math.max.apply(null, timeline.map(function (p) { return p.live_bytes || 0; }).concat([1]));
    var w = 800;
    var h = 200;
    var pad = 32;
    html += '<svg class="memory-chart" viewBox="0 0 ' + w + " " + h + '" role="img" aria-label="Memory timeline">';
    html += '<polyline fill="none" stroke="var(--accent)" stroke-width="2" points="';
    html += timeline.map(function (p) {
      var x = pad + (p.timestamp_ns / maxTs) * (w - pad * 2);
      var y = h - pad - ((p.live_bytes || 0) / maxLive) * (h - pad * 2);
      return x.toFixed(1) + "," + y.toFixed(1);
    }).join(" ");
    html += '"/>';
    html += '<text x="' + pad + '" y="' + (h - 8) + '" fill="var(--muted)" font-size="10">0</text>';
    html += '<text x="' + (w - pad) + '" y="' + (h - 8) + '" fill="var(--muted)" font-size="10" text-anchor="end">' + esc(fmtNsMs(maxTs)) + "</text>";
    html += '<text x="8" y="' + pad + '" fill="var(--muted)" font-size="10">' + esc(fmtBytes(maxLive)) + "</text>";
    html += "</svg>";
    html += '<div class="table-wrap"><table class="timeline-table"><caption class="sr">Memory timeline events</caption><thead><tr><th scope="col">Time</th><th scope="col">Device</th><th scope="col">Live</th><th scope="col">Heap</th></tr></thead><tbody>';
    html += timeline.slice(-100).map(function (p) {
      return "<tr><td>" + esc(fmtNsMs(p.timestamp_ns)) + "</td><td>" + esc(p.device || "") +
        "</td><td>" + esc(fmtBytes(p.live_bytes)) + "</td><td>" + esc(fmtBytes(p.heap_bytes)) + "</td></tr>";
    }).join("");
    html += "</tbody></table></div>";
    panel.innerHTML = html;
  }

  function renderSpanCosts(data) {
    var panel = panelFor("span_costs");
    if (!panel) return;
    var items = data.items || [];
    if (!items.length) {
      panel.innerHTML = '<div class="view-heading"><div><p class="eyebrow">Aggregated semantic work</p><h1>Span costs</h1></div></div>' +
        '<div class="content-empty" role="status"><strong>No span costs</strong><p>This profile contains no aggregated span timing rows.</p></div>';
      return;
    }
    panel.innerHTML =
      '<div class="view-heading"><div><p class="eyebrow">Aggregated semantic work</p><h1>Span costs</h1></div><p>' +
      items.length + ' measured spans</p></div><div class="table-wrap"><table class="timeline-table span-cost-table"><caption class="sr">Span cost ranking</caption>' +
      '<thead><tr><th scope="col">Span</th><th scope="col">Kind</th><th scope="col">Self</th><th scope="col">Total</th><th scope="col">Memory</th></tr></thead><tbody>' +
      items.map(function (it) {
        return '<tr data-id="' + esc(idStr(it.id)) + '"><td><button type="button" class="table-row-action" data-span-cost-id="' +
          esc(idStr(it.id)) + '">' + esc(it.name) + '</button></td><td>' + esc(it.kind || "") +
          "</td><td>" + esc(fmtMs(it.self_ms)) + "</td><td>" + esc(fmtMs(it.total_ms)) +
          "</td><td>" + esc(fmtBytes(it.peak_bytes || it.bytes)) + "</td></tr>";
      }).join("") +
      "</tbody></table></div>";
    panel.querySelectorAll("[data-span-cost-id]").forEach(function (button) {
      button.onclick = function () {
        var row = button.closest("tr");
        panel.querySelectorAll("tbody tr.sel").forEach(function (x) { x.classList.remove("sel"); });
        row.classList.add("sel");
        selectedId = row.dataset.id;
        var it = items.find(function (x) { return idStr(x.id) === selectedId; });
        setInspector(it);
      };
      button.onkeydown = function (event) {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        button.click();
      };
    });
  }

  function renderGpuView(data) {
    var panel = panelFor("gpu");
    if (!panel) return;
    var status = data.status || (data.available ? "available" : "unavailable");
    var available = data.available === true || ["available", "captured", "complete"].includes(safeStatus(status));
    var sources = Array.isArray(data.source_csv) ? data.source_csv : (data.source_csv ? [data.source_csv] : []);
    var sourceHtml = '<dl class="key-values source-list"><div><dt>Raw report</dt><dd>' + esc(data.raw_report || "Not retained") +
      '</dd></div><div><dt>Normalized CSV</dt><dd>' + esc(sources.length ? sources.join(", ") : "Not available") + "</dd></div></dl>";
    var html = '<div class="view-heading"><div><p class="eyebrow">Nsight Systems correlation</p><h1>GPU evidence</h1></div>' +
      statusLabel(status) + "</div>";
    var trustHtml = '<section class="evidence-card gpu-sources"><h2>Coverage and correlation trust</h2>' +
      renderKeyValues({ coverage: data.coverage, correlation: data.correlation, limits: data.limits }) +
      renderNoticeList(data.diagnostics, "warning", "No normalization diagnostics.") + "</section>";
    if (!available) {
      html += '<div class="gpu-empty" role="status"><div class="gpu-empty-mark" aria-hidden="true">GPU</div>' +
        '<div><h2>GPU evidence is not available</h2><p>' + esc(data.reason || "This profile was captured without Nsight Systems evidence.") +
        "</p><p>Candle span and memory evidence remain available in the other views.</p></div></div>" + sourceHtml + trustHtml;
      panel.innerHTML = html;
      return;
    }
    html += '<section class="evidence-card gpu-sources"><h2>Capture sources</h2>' + sourceHtml + "</section>" + trustHtml;
    [
      ["NVTX projected ranges", data.nvtx_ranges],
      ["CUDA kernels", data.kernels],
      ["CUDA runtime calls", data.runtime_calls],
      ["GPU memory operations", data.memory_operations],
      ["GPU timeline", data.gpu_timeline],
    ].forEach(function (section) {
      html += '<section class="data-section"><h2>' + esc(section[0]) + "</h2>" + renderDataTable(section[1], section[0], 250) + "</section>";
    });
    panel.innerHTML = html;
  }

  function buildSpanTree() {
    var tree = document.getElementById("span-tree");
    if (!tree) return;
    var spans = P.span_tree || [];
    var rootKey = "__candle_graph_root__";
    var q = (document.querySelector("[data-span-search]") || {}).value || "";
    q = q.trim().toLowerCase();
    var byParent = {};
    spans.forEach(function (s) {
      var p = s.parent_id == null ? rootKey : idStr(s.parent_id);
      (byParent[p] = byParent[p] || []).push(s);
    });
    Object.keys(byParent).forEach(function (k) {
      byParent[k].sort(function (a, b) { return (b.total_ms || 0) - (a.total_ms || 0); });
    });
    if (!spanOpen.size && byParent[rootKey]) {
      byParent[rootKey].forEach(function (s) { spanOpen.add(idStr(s.id)); });
    }

    function render() {
      tree.innerHTML = "";
      function add(list, depth) {
        (list || []).forEach(function (s) {
          var id = idStr(s.id);
          if (q && !String(s.name || "").toLowerCase().includes(q)) return;
          var kids = byParent[id] || [];
          var has = kids.length > 0;
          var expanded = spanOpen.has(id);
          var row = document.createElement("div");
          row.className = "span-row";
          row.setAttribute("role", "treeitem");
          row.setAttribute("aria-level", String(depth + 1));
          row.style.paddingLeft = depth * 14 + 8 + "px";
          row.dataset.spanId = id;
          row.tabIndex = selectedId === id ? 0 : -1;
          row.setAttribute("aria-selected", selectedId === id ? "true" : "false");
          if (has) row.setAttribute("aria-expanded", expanded ? "true" : "false");
          if (has) {
            var b = document.createElement("button");
            b.type = "button";
            b.className = "tw";
            b.textContent = expanded ? "▾" : "▸";
            b.tabIndex = -1;
            b.setAttribute("aria-label", (expanded ? "Collapse " : "Expand ") + s.name);
            b.onclick = function (e) {
              e.stopPropagation();
              if (spanOpen.has(id)) spanOpen.delete(id); else spanOpen.add(id);
              render();
              var restored = findSpanRow(tree, id);
              if (restored) restored.focus();
            };
            row.appendChild(b);
          } else {
            var sp = document.createElement("span");
            sp.className = "tw";
            sp.textContent = "·";
            row.appendChild(sp);
          }
          var name = document.createElement("span");
          name.className = "span-name";
          name.textContent = s.name;
          name.style.color = heatColor(s);
          row.appendChild(name);
          var ms = document.createElement("span");
          ms.className = "span-ms";
          ms.textContent = fmtMs(s.self_ms) + " / " + fmtMs(s.total_ms) +
            (s.peak_bytes || s.bytes ? " · " + fmtBytes(s.peak_bytes || s.bytes) : "");
          row.appendChild(ms);
          function activate() {
            tree.querySelectorAll("[aria-selected=true]").forEach(function (x) {
              x.setAttribute("aria-selected", "false");
              x.tabIndex = -1;
            });
            row.setAttribute("aria-selected", "true");
            row.tabIndex = 0;
            selectedId = id;
            setInspector(Object.assign({ label: s.name }, s));
            highlightNode(id);
          }
          row.onclick = activate;
          row.onkeydown = function (event) {
            var rows = Array.from(tree.querySelectorAll("[data-span-id]"));
            var index = rows.indexOf(row);
            var target = null;
            if (event.key === "Enter" || event.key === " ") activate();
            else if (event.key === "ArrowDown") target = rows[Math.min(rows.length - 1, index + 1)];
            else if (event.key === "ArrowUp") target = rows[Math.max(0, index - 1)];
            else if (event.key === "Home") target = rows[0];
            else if (event.key === "End") target = rows[rows.length - 1];
            else if (event.key === "ArrowRight" && has) {
              if (!spanOpen.has(id)) {
                spanOpen.add(id);
                render();
                target = findSpanRow(tree, id);
              } else {
                target = rows[index + 1];
              }
            } else if (event.key === "ArrowLeft") {
              if (has && spanOpen.has(id)) {
                spanOpen.delete(id);
                render();
                target = findSpanRow(tree, id);
              } else if (s.parent_id != null) {
                target = findSpanRow(tree, idStr(s.parent_id));
              }
            } else return;
            event.preventDefault();
            if (target) target.focus();
          };
          tree.appendChild(row);
          if (has && expanded) add(kids, depth + 1);
        });
      }
      add(byParent[rootKey], 0);
      if (!tree.querySelector('[tabindex="0"]')) {
        var first = tree.querySelector("[data-span-id]");
        if (first) first.tabIndex = 0;
      }
    }
    var search = document.querySelector("[data-span-search]");
    if (search) search.oninput = render;
    render();
  }

  function highlightNode(id) {
    selectedId = id;
    if (graphState) graphState.updateHighlight();
    document.querySelectorAll("#span-tree [data-span-id]").forEach(function (el) {
      var selected = el.dataset.spanId === id;
      el.setAttribute("aria-selected", selected ? "true" : "false");
      el.tabIndex = selected ? 0 : -1;
    });
  }

  function svgPoint(svg, clientX, clientY) {
    var r = svg.getBoundingClientRect();
    return { x: clientX - r.left, y: clientY - r.top };
  }

  function fitView(vis) {
    if (!vis.length) return;
    var wrap = document.getElementById("view-panel-trace");
    var pw = wrap.clientWidth || 800;
    var ph = wrap.clientHeight || 400;
    var coords = vis.filter(function (n) { return Number.isFinite(n._x) && Number.isFinite(n._y); });
    if (!coords.length) return;
    var pad = 64;
    var minX = Math.min.apply(null, coords.map(function (n) { return n._x; }));
    var maxX = Math.max.apply(null, coords.map(function (n) { return n._x + (n._w || 0); }));
    var minY = Math.min.apply(null, coords.map(function (n) { return n._y; }));
    var maxY = Math.max.apply(null, coords.map(function (n) { return n._y + (n._h || 0); }));
    var gw = Math.max(maxX - minX + pad * 2, 1);
    var gh = Math.max(maxY - minY + pad * 2, 1);
    graphView.k = Math.min(2.5, Math.max(0.06, Math.min(pw / gw, ph / gh)));
    var cx = (minX + maxX) / 2;
    var cy = (minY + maxY) / 2;
    graphView.x = pw / 2 - cx * graphView.k;
    graphView.y = ph / 2 - cy * graphView.k;
    updateZoomLabel();
  }

  function zoomAt(sx, sy, factor) {
    var k0 = graphView.k;
    var k1 = Math.min(4, Math.max(0.06, k0 * factor));
    var gx = (sx - graphView.x) / k0;
    var gy = (sy - graphView.y) / k0;
    graphView.k = k1;
    graphView.x = sx - gx * k1;
    graphView.y = sy - gy * k1;
    updateZoomLabel();
  }

  function updateZoomLabel() {
    var el = document.getElementById("zoom-label");
    if (el) el.textContent = Math.round(graphView.k * 100) + "%";
  }

  function neighborSet(nodeId, edges) {
    var s = new Set([nodeId]);
    edges.forEach(function (e) {
      if (e._from === nodeId) s.add(e._to);
      if (e._to === nodeId) s.add(e._from);
    });
    return s;
  }

  var tooltip = document.getElementById("graph-tooltip");

  function showTooltip(n, clientX, clientY) {
    if (!tooltip || !n) return;
    var wrap = document.getElementById("view-panel-trace");
    var r = wrap.getBoundingClientRect();
    tooltip.innerHTML =
      '<div class="tt-title">' + esc(n.label || n.name || "") + "</div>" +
      '<div class="tt-meta">self ' + esc(fmtMs(n.self_time_ms)) + " · total " + esc(fmtMs(n.total_time_ms)) +
      (n.peak_bytes || n.bytes ? " · " + esc(fmtBytes(n.peak_bytes || n.bytes)) : "") + "</div>";
    tooltip.classList.add("visible");
    var tx = Math.min(clientX - r.left + 12, r.width - tooltip.offsetWidth - 8);
    var ty = Math.min(clientY - r.top + 12, r.height - tooltip.offsetHeight - 8);
    tooltip.style.left = Math.max(8, tx) + "px";
    tooltip.style.top = Math.max(8, ty) + "px";
  }

  function hideTooltip() {
    if (tooltip) tooltip.classList.remove("visible");
  }

  function buildNodeBody(n) {
    var kind = n.kind || "function";
    var lines = n._titleLines || [CGLayout.labelOf(n)];
    var sub = n._sub || (fmtMs(n.self_time_ms) + " self · " + fmtMs(n.total_time_ms) + " total" +
      (n.peak_bytes || n.bytes ? " · " + fmtBytes(n.peak_bytes || n.bytes) : ""));
    var html = '<div class="nb-head"><span class="nb-kind">' + esc(kind) + "</span></div>";
    html += lines.map(function (l) { return '<div class="nb-title">' + esc(l) + "</div>"; }).join("");
    html += '<div class="nb-sub">' + esc(sub) + "</div>";
    return html;
  }

  function drawGraph(data, meta) {
    var layout = meta.layout || "layered";
    var direction = meta.direction || "LR";
    var svg = document.getElementById("graph-canvas");
    var empty = document.getElementById("empty-graph");
    var NS = svg.namespaceURI;
    var XHTML = "http://www.w3.org/1999/xhtml";

    var nodes = (data.nodes || []).map(function (n, i) {
      return Object.assign({}, n, { _id: idStr(n.id != null ? n.id : i) });
    });
    if (!nodes.length) {
      svg.innerHTML = "";
      if (empty) empty.hidden = false;
      graphState = null;
      return;
    }
    if (empty) empty.hidden = true;
    svg.removeAttribute("aria-hidden");

    var edges = (data.edges || []).map(function (e, i) {
      return Object.assign({}, e, {
        _id: String(e.id != null ? e.id : "e" + i),
        _from: idStr(e.from),
        _to: idStr(e.to),
        label: e.label || (e.duration_ms != null ? fmtMs(e.duration_ms) : ""),
      });
    });

    document.querySelector("[data-graph-stats]").textContent = nodes.length + " spans · " + edges.length + " edges";

    if (layout === "tree") CGLayout.layoutTree(nodes, edges);
    else CGLayout.layoutLayered(nodes, edges, direction);

    var byId = {};
    nodes.forEach(function (n) { byId[n._id] = n; });
    var visEdges = edges.filter(function (e) { return byId[e._from] && byId[e._to]; });
    CGLayout.assignEdgePorts(nodes, visEdges, byId, layout, direction);

    var rootG = document.createElementNS(NS, "g");
    var bandG = document.createElementNS(NS, "g");
    var edgeG = document.createElementNS(NS, "g");
    var nodeG = document.createElementNS(NS, "g");
    rootG.appendChild(bandG);
    rootG.appendChild(edgeG);
    rootG.appendChild(nodeG);

    function focusSet() {
      var id = hoveredId || selectedId;
      if (!id) return null;
      return neighborSet(id, visEdges);
    }

    function updateClasses() {
      var focus = focusSet();
      nodeG.querySelectorAll(".node").forEach(function (el) {
        var id = el.dataset.nodeId;
        el.classList.toggle("dim", focus && !focus.has(id) && id !== selectedId);
        el.classList.toggle("sel", id === selectedId);
      });
      edgeG.querySelectorAll(".edge-group").forEach(function (el) {
        var lit = focus && (focus.has(el.dataset.from) || focus.has(el.dataset.to));
        el.classList.toggle("dim", focus && !lit);
        el.classList.toggle("sel", el.dataset.edgeId === selectedId);
      });
    }

    function buildSVG() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      rootG.setAttribute("transform", "translate(" + graphView.x + "," + graphView.y + ") scale(" + graphView.k + ")");
      svg.appendChild(rootG);
      bandG.innerHTML = "";
      edgeG.innerHTML = "";
      nodeG.innerHTML = "";

      CGLayout.layerBands(nodes, direction).forEach(function (b, i) {
        var rect = document.createElementNS(NS, "rect");
        rect.setAttribute("class", "layer-band");
        rect.setAttribute("x", b.x);
        rect.setAttribute("y", b.y);
        rect.setAttribute("width", b.w);
        rect.setAttribute("height", b.h);
        rect.setAttribute("rx", "12");
        if (i % 2) rect.setAttribute("opacity", "0.55");
        bandG.appendChild(rect);
      });

      visEdges.forEach(function (e) {
        var edgeKind = e.kind === "data" ? "edge-composition" : "edge-default";
        var pathD = CGLayout.routeEdge(e);
        var g = document.createElementNS(NS, "g");
        g.setAttribute("class", "edge-group");
        g.dataset.from = e._from;
        g.dataset.to = e._to;
        g.dataset.edgeId = e._id;

        var hit = document.createElementNS(NS, "path");
        hit.setAttribute("class", "edge-hit");
        hit.setAttribute("d", pathD);
        hit.setAttribute("fill", "none");
        hit.setAttribute("stroke", "transparent");
        hit.setAttribute("stroke-width", "14");

        var p = document.createElementNS(NS, "path");
        p.setAttribute("class", "edge " + edgeKind);
        p.setAttribute("stroke-width", "2");
        p.setAttribute("fill", "none");
        p.setAttribute("marker-end", "url(#arrow)");
        p.setAttribute("d", pathD);

        var activate = function (ev) {
          ev.stopPropagation();
          selectedId = e._id;
          setInspector(Object.assign({ kind: "edge", label: e.label }, e));
          updateClasses();
        };
        hit.onclick = activate;
        p.onclick = activate;
        g.appendChild(hit);
        g.appendChild(p);

        var label = e.label || (e.duration_ms != null ? fmtMs(e.duration_ms) : "");
        if (label) {
          var mid = CGLayout.edgeMidpoint(e);
          if (mid) {
            var lbl = document.createElementNS(NS, "text");
            lbl.setAttribute("class", "edge-label");
            lbl.setAttribute("x", mid.x);
            lbl.setAttribute("y", mid.y);
            lbl.setAttribute("text-anchor", "middle");
            lbl.textContent = label;
            g.appendChild(lbl);
          }
        }
        edgeG.appendChild(g);
      });

      nodes.forEach(function (n) {
        var gg = document.createElementNS(NS, "g");
        var kind = n.kind || "function";
        gg.setAttribute("class", "node " + kind);
        gg.dataset.nodeId = n._id;
        gg.setAttribute("tabindex", "0");
        gg.setAttribute("role", "button");
        gg.setAttribute("aria-label", (n.label || n.name || "Span") + ", self " +
          fmtMs(n.self_time_ms) + ", total " + fmtMs(n.total_time_ms));

        var card = document.createElementNS(NS, "rect");
        card.setAttribute("class", "node-card");
        card.setAttribute("x", n._x);
        card.setAttribute("y", n._y);
        card.setAttribute("width", n._w);
        card.setAttribute("height", n._h);
        card.setAttribute("rx", "8");
        card.setAttribute("stroke", heatColor(n));
        gg.appendChild(card);

        var fo = document.createElementNS(NS, "foreignObject");
        fo.setAttribute("x", n._x);
        fo.setAttribute("y", n._y);
        fo.setAttribute("width", n._w);
        fo.setAttribute("height", n._h);
        var div = document.createElementNS(XHTML, "div");
        div.setAttribute("class", "node-body");
        div.innerHTML = buildNodeBody(n);
        fo.appendChild(div);
        gg.appendChild(fo);

        gg.onmouseenter = function (ev) {
          hoveredId = n._id;
          updateClasses();
          showTooltip(n, ev.clientX, ev.clientY);
        };
        gg.onmouseleave = function () {
          if (hoveredId === n._id) hoveredId = null;
          updateClasses();
          hideTooltip();
        };
        gg.onmousemove = function (ev) { showTooltip(n, ev.clientX, ev.clientY); };
        gg.onclick = function (ev) {
          ev.stopPropagation();
          selectedId = n._id;
          setInspector(n);
          updateClasses();
          document.querySelectorAll("#span-tree [data-span-id]").forEach(function (el) {
            var selected = el.dataset.spanId === n._id;
            el.setAttribute("aria-selected", selected ? "true" : "false");
            el.tabIndex = selected ? 0 : -1;
          });
        };
        gg.onkeydown = function (ev) {
          if (ev.key !== "Enter" && ev.key !== " ") return;
          ev.preventDefault();
          gg.onclick(ev);
        };
        nodeG.appendChild(gg);
      });

      updateClasses();
    }

    function ensureDefs() {
      var defs = svg.querySelector("defs");
      if (!defs) {
        defs = document.createElementNS(NS, "defs");
        svg.insertBefore(defs, svg.firstChild);
      }
      if (!svg.querySelector("#arrow")) {
        var m = document.createElementNS(NS, "marker");
        m.setAttribute("id", "arrow");
        m.setAttribute("viewBox", "0 0 10 10");
        m.setAttribute("refX", "9");
        m.setAttribute("refY", "5");
        m.setAttribute("markerWidth", "6");
        m.setAttribute("markerHeight", "6");
        m.setAttribute("orient", "auto-start-reverse");
        var path = document.createElementNS(NS, "path");
        path.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
        path.setAttribute("fill", "var(--edge)");
        m.appendChild(path);
        defs.appendChild(m);
      }
    }

    ensureDefs();
    buildSVG();
    if (graphView._fit) {
      fitView(nodes);
      graphView._fit = false;
    }

    graphState = {
      nodes: nodes,
      applyView: buildSVG,
      updateHighlight: updateClasses,
    };
  }

  function initCanvasControls() {
    var wrap = document.getElementById("view-panel-trace");
    document.getElementById("fit-btn").onclick = function () {
      if (!graphState) return;
      fitView(graphState.nodes);
      graphState.applyView();
    };
    document.getElementById("reset-btn").onclick = function () {
      graphView = { x: 0, y: 0, k: 1, _fit: true };
      refreshView();
    };
    document.getElementById("zoom-in").onclick = function () {
      if (!graphState) return;
      zoomAt(wrap.clientWidth / 2, wrap.clientHeight / 2, 1.2);
      graphState.applyView();
    };
    document.getElementById("zoom-out").onclick = function () {
      if (!graphState) return;
      zoomAt(wrap.clientWidth / 2, wrap.clientHeight / 2, 1 / 1.2);
      graphState.applyView();
    };
    document.getElementById("zoom-fit").onclick = function () {
      if (!graphState) return;
      fitView(graphState.nodes);
      graphState.applyView();
    };
    wrap.addEventListener(
      "wheel",
      function (e) {
        if (!graphState || currentView !== "trace") return;
        e.preventDefault();
        var pt = svgPoint(document.getElementById("graph-canvas"), e.clientX, e.clientY);
        zoomAt(pt.x, pt.y, e.deltaY < 0 ? 1.08 : 1 / 1.08);
        graphState.applyView();
      },
      { passive: false }
    );
  }

  function initTraceUtilities() {
    var legend = document.querySelector("[data-legend]");
    var legendToggle = document.getElementById("legend-toggle");
    if (legend && legendToggle) {
      legendToggle.onclick = function () {
        var collapsed = legend.classList.toggle("collapsed");
        legendToggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
      };
    }
    var exportButton = document.getElementById("export-btn");
    if (exportButton) {
      exportButton.onclick = function () {
        var svg = document.getElementById("graph-canvas");
        if (!svg || !svg.childNodes.length) return;
        var clone = svg.cloneNode(true);
        clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
        var blob = new Blob([new XMLSerializer().serializeToString(clone)], { type: "image/svg+xml" });
        var href = URL.createObjectURL(blob);
        var link = document.createElement("a");
        link.href = href;
        link.download = "candle-graph-trace.svg";
        link.click();
        setTimeout(function () { URL.revokeObjectURL(href); }, 0);
      };
    }
  }

  document.getElementById("theme-btn").onclick = function () {
    var n = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", n);
    localStorage.setItem("cg-theme", n);
  };

  document.querySelectorAll('input[name="heat-mode"]').forEach(function (input) {
    input.addEventListener("change", function () {
      if (!input.checked) return;
      heatMode = input.value;
      if (graphState) graphState.applyView();
      buildSpanTree();
    });
  });

  initResizers();
  initTabs();
  initCanvasControls();
  initTraceUtilities();
  renderCoverage();
  renderPeakBreakdown();
  buildSpanTree();
  setInspector(null);
  graphView._fit = true;
  selectView(currentView, false);
})();
