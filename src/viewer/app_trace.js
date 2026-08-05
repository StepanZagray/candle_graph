(function () {
  "use strict";

  const P = JSON.parse(document.getElementById("cg-payload").textContent);
  const V = P.views || {};
  const SUM = P.summary || {};

  const VIEW_META = [
    { id: "trace", label: "Trace", graph: true, layout: "layered", direction: "LR" },
    { id: "timeline", label: "Timeline", graph: false },
    { id: "memory", label: "Memory", graph: false },
  ];

  let currentView = P.default_view || "trace";
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
      panel.innerHTML = "<p class=\"empty\">No peak allocations recorded.</p>";
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
      return '<button type="button" role="tab" class="tab" data-view="' + esc(m.id) + '" aria-selected="false">' + esc(m.label) + "</button>";
    }).join("");
    tabs.onclick = function (e) {
      var btn = e.target.closest("[data-view]");
      if (!btn) return;
      selectView(btn.dataset.view, true);
    };
  }

  function selectView(id, push) {
    currentView = id;
    document.querySelectorAll("[data-view-tabs] [data-view]").forEach(function (b) {
      b.setAttribute("aria-selected", b.dataset.view === id ? "true" : "false");
    });
    var meta = VIEW_META.find(function (m) { return m.id === id; }) || VIEW_META[0];
    var title = document.querySelector("[data-canvas-title]");
    if (title) title.textContent = meta.label;
    refreshView();
    if (push) setInspector(null);
  }

  function refreshView() {
    var meta = VIEW_META.find(function (m) { return m.id === currentView; }) || VIEW_META[0];
    var svg = document.getElementById("graph-canvas");
    var timeline = document.getElementById("timeline-panel");
    var empty = document.getElementById("empty-graph");
    if (meta.graph) {
      if (svg) svg.hidden = false;
      if (timeline) timeline.hidden = true;
      drawGraph(V.trace || { nodes: [], edges: [] }, meta);
    } else if (meta.id === "memory") {
      if (svg) { svg.hidden = true; svg.innerHTML = ""; }
      if (timeline) timeline.hidden = false;
      graphState = null;
      renderMemoryView(V.memory || { timeline: [], peak_breakdown: [], summary: {} });
      if (empty) empty.hidden = true;
    } else {
      if (svg) { svg.hidden = true; svg.innerHTML = ""; }
      if (timeline) timeline.hidden = false;
      graphState = null;
      renderTimeline(V.timeline || { items: [] });
      if (empty) empty.hidden = (V.timeline && V.timeline.items && V.timeline.items.length) > 0;
    }
  }

  function renderMemoryView(data) {
    var panel = document.getElementById("timeline-panel");
    if (!panel) return;
    var timeline = data.timeline || [];
    var summary = data.summary || {};
    var html = "";
    html += "<div class=\"pane-h sub\">Memory summary</div>";
    html += "<p style=\"font-size:12px;margin:0 0 8px;color:var(--muted)\">";
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
      html += "<p class=\"empty\">No memory timeline events.</p>";
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
    html += '<table class="timeline-table"><thead><tr><th>Time</th><th>Device</th><th>Live</th><th>Heap</th></tr></thead><tbody>';
    html += timeline.slice(-100).map(function (p) {
      return "<tr><td>" + esc(fmtNsMs(p.timestamp_ns)) + "</td><td>" + esc(p.device || "") +
        "</td><td>" + esc(fmtBytes(p.live_bytes)) + "</td><td>" + esc(fmtBytes(p.heap_bytes)) + "</td></tr>";
    }).join("");
    html += "</tbody></table>";
    panel.innerHTML = html;
  }

  function renderTimeline(data) {
    var panel = document.getElementById("timeline-panel");
    if (!panel) return;
    var items = data.items || [];
    if (!items.length) {
      panel.innerHTML = "<p class=\"empty\">No timeline items.</p>";
      return;
    }
    panel.innerHTML =
      '<table class="timeline-table"><thead><tr><th>Span</th><th>Kind</th><th>Self</th><th>Total</th><th>Memory</th></tr></thead><tbody>' +
      items.map(function (it) {
        return '<tr data-id="' + esc(idStr(it.id)) + '"><td>' + esc(it.name) + '</td><td>' + esc(it.kind || "") +
          "</td><td>" + esc(fmtMs(it.self_ms)) + "</td><td>" + esc(fmtMs(it.total_ms)) +
          "</td><td>" + esc(fmtBytes(it.peak_bytes || it.bytes)) + "</td></tr>";
      }).join("") +
      "</tbody></table>";
    panel.querySelectorAll("tbody tr").forEach(function (row) {
      row.onclick = function () {
        panel.querySelectorAll("tbody tr.sel").forEach(function (x) { x.classList.remove("sel"); });
        row.classList.add("sel");
        selectedId = row.dataset.id;
        var it = items.find(function (x) { return idStr(x.id) === selectedId; });
        setInspector(it);
        if (currentView === "trace") highlightNode(selectedId);
      };
    });
  }

  function buildSpanTree() {
    var tree = document.getElementById("span-tree");
    if (!tree) return;
    var spans = P.span_tree || [];
    var q = (document.querySelector("[data-span-search]") || {}).value || "";
    q = q.trim().toLowerCase();
    var byParent = {};
    spans.forEach(function (s) {
      var p = s.parent_id == null ? "root" : idStr(s.parent_id);
      (byParent[p] = byParent[p] || []).push(s);
    });
    Object.keys(byParent).forEach(function (k) {
      byParent[k].sort(function (a, b) { return (b.total_ms || 0) - (a.total_ms || 0); });
    });
    if (!spanOpen.size && byParent.root) {
      byParent.root.forEach(function (s) { spanOpen.add(idStr(s.id)); });
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
          row.style.paddingLeft = depth * 14 + 8 + "px";
          row.dataset.spanId = id;
          if (selectedId === id) row.setAttribute("aria-selected", "true");
          if (has) {
            var b = document.createElement("button");
            b.type = "button";
            b.className = "tw";
            b.textContent = expanded ? "▾" : "▸";
            b.onclick = function (e) {
              e.stopPropagation();
              if (spanOpen.has(id)) spanOpen.delete(id); else spanOpen.add(id);
              render();
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
          row.onclick = function () {
            tree.querySelectorAll("[aria-selected=true]").forEach(function (x) {
              x.setAttribute("aria-selected", "false");
            });
            row.setAttribute("aria-selected", "true");
            selectedId = id;
            setInspector(Object.assign({ label: s.name }, s));
            highlightNode(id);
          };
          tree.appendChild(row);
          if (has && expanded) add(kids, depth + 1);
        });
      }
      add(byParent.root, 0);
    }
    var search = document.querySelector("[data-span-search]");
    if (search) search.oninput = render;
    render();
  }

  function highlightNode(id) {
    selectedId = id;
    if (graphState) graphState.updateHighlight();
    document.querySelectorAll("#span-tree [data-span-id]").forEach(function (el) {
      el.setAttribute("aria-selected", el.dataset.spanId === id ? "true" : "false");
    });
  }

  function svgPoint(svg, clientX, clientY) {
    var r = svg.getBoundingClientRect();
    return { x: clientX - r.left, y: clientY - r.top };
  }

  function fitView(vis) {
    if (!vis.length) return;
    var wrap = document.getElementById("canvas-wrap");
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
    var wrap = document.getElementById("canvas-wrap");
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
            el.setAttribute("aria-selected", el.dataset.spanId === n._id ? "true" : "false");
          });
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
    var wrap = document.getElementById("canvas-wrap");
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
  renderCoverage();
  renderPeakBreakdown();
  buildSpanTree();
  setInspector(null);
  graphView._fit = true;
  selectView(currentView, false);
})();
