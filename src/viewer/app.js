(function () {
  "use strict";

  const P = JSON.parse(document.getElementById("cg-payload").textContent);
  const V = P.views || {};
  const GRAD = ["Trainable", "Frozen", "Differentiable", "Severed", "LayoutDependent", "Unknown"];

  const VIEW_META = [
    { id: "architecture", label: "Architecture", graph: true, layout: "tree" },
    { id: "dataflow_train", label: "Train", graph: true, layout: "layered", scoped: true, direction: "LR" },
    { id: "dataflow_infer", label: "Infer", graph: true, layout: "layered", scoped: true, direction: "LR" },
    { id: "pipeline", label: "Pipeline", graph: true, layout: "layered", direction: "TB" },
    { id: "findings", label: "Findings", graph: false },
  ];

  const KIND_LABEL = {
    operation: "Op",
    parameter: "Param",
    tensor: "Tensor",
    component: "Component",
    module: "Module",
    stage: "Stage",
  };

  let currentView = "architecture";
  let highlights = new Set();
  let selectedId = null;
  let hoveredId = null;
  let selectedFunction = "";
  let graphState = null;
  let graphView = { x: 0, y: 0, k: 1 };

  const root = document.documentElement;
  const pref = matchMedia("(prefers-color-scheme:dark)").matches ? "dark" : "light";
  root.setAttribute("data-theme", localStorage.getItem("cg-theme") || pref);

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch])
    );
  }
  function idStr(v) { return v && (v.id || v._id || String(v)); }
  function fnId(v) { return v == null ? "" : String(v.id != null ? v.id : v); }
  function fmtShape(s) {
    if (s == null) return "—";
    return Array.isArray(s) ? s.join(" × ") : String(s);
  }
  function fmtTiming(o) {
    if (o.timing) {
      var t = o.timing;
      var avg = (t.avg_ns / 1e6).toFixed(3);
      var lo = (t.min_ns / 1e6).toFixed(3);
      var hi = (t.max_ns / 1e6).toFixed(3);
      return avg + " ms avg · " + lo + "–" + hi + " ms (" + t.samples + " samples)";
    }
    if (o.avg_duration_ms != null) return o.avg_duration_ms.toFixed(3) + " ms";
    if (o.avg_duration_ns != null) return (o.avg_duration_ns / 1e6).toFixed(3) + " ms";
    return "—";
  }

  function initResizers() {
    const sidebar = document.querySelector(".pane-sidebar");
    const inspector = document.querySelector(".pane-inspector");
    document.querySelectorAll(".resize-handle").forEach((handle) => {
      handle.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        handle.classList.add("active");
        const side = handle.dataset.side;
        const startX = e.clientX;
        const startW = side === "left" ? sidebar.offsetWidth : inspector.offsetWidth;
        function move(ev) {
          const dx = ev.clientX - startX;
          if (side === "left") {
            const w = Math.max(200, Math.min(window.innerWidth * 0.45, startW + dx));
            root.style.setProperty("--sidebar-w", w + "px");
          } else {
            const w = Math.max(200, Math.min(window.innerWidth * 0.45, startW - dx));
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
    const c = P.coverage || {};
    const s = P.summary || {};
    const el = document.querySelector("[data-coverage]");
    el.textContent = [
      P.package || s.model_name,
      c.components != null ? c.components + " components" : null,
      c.modules != null ? c.modules + " modules" : null,
      c.parameters != null ? c.parameters + " params" : null,
      s.trainable_parameters != null ? s.trainable_parameters + " trainable" : null,
      c.operations != null ? c.operations + " ops" : null,
      c.diagnostics != null ? c.diagnostics + " findings" : null,
    ].filter(Boolean).join(" · ") || "No coverage";

    const box = document.querySelector("[data-diagnostics]");
    const diags = P.diagnostics || [];
    box.innerHTML = diags.length
      ? diags.map((d) => `<div>${esc(d.at ? d.at + " — " : "")}${esc(d.message || d)}</div>`).join("")
      : "";
  }

  function setInspector(o) {
    const empty = !o || (!o.label && !o.key && !o.name && !o.rule && o.kind !== "edge");
    const insp = document.getElementById("inspector");
    if (insp) insp.classList.toggle("is-empty", empty);
    o = o || {};
    const fields = {
      source: o.source || o.at || "—",
      shape: fmtShape(o.shape),
      dtype: o.dtype || "—",
      root: o.builder_root || o.root || "—",
      confidence: o.confidence || "—",
      grad: o.grad_state || o.grad || "—",
      label: empty ? "Nothing selected" : (o.label || o.key || o.name || o.type_name || "—"),
      kind: o.kind || o.op || "—",
      severity: o.severity || "—",
      rule: o.rule || "—",
      qualified: o.qualified_name || "—",
      role: o.role || "—",
      timing: fmtTiming(o),
      function: o.function ? fnId(o.function) : "—",
    };
    Object.keys(fields).forEach((k) => {
      const el = document.querySelector(`[data-field="${k}"]`);
      if (el) el.textContent = fields[k];
    });
  }

  function clearSelection(clearHighlights) {
    selectedId = null;
    hoveredId = null;
    hideTooltip();
    if (clearHighlights) highlights.clear();
    setInspector(null);
    document.querySelectorAll("#module-tree [aria-selected=true]").forEach((el) => {
      el.setAttribute("aria-selected", "false");
    });
    if (graphState) graphState.updateHighlight();
  }

  function isGraphItemTarget(target) {
    return !!(target && target.closest && target.closest(".node,.edge-group,.edge-hit,.edge"));
  }

  function initTabs() {
    const tabs = document.querySelector("[data-view-tabs]");
    tabs.innerHTML = "";
    VIEW_META.forEach((v) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "tab";
      b.setAttribute("role", "tab");
      b.dataset.view = v.id;
      b.textContent = v.label;
      b.setAttribute("aria-selected", v.id === currentView ? "true" : "false");
      b.onclick = () => selectView(v.id);
      tabs.appendChild(b);
    });
  }

  function viewData(id) {
    return V[id] || { nodes: [], edges: [] };
  }

  function defaultFunction(data) {
    const fns = data.functions || [];
    const forward = fns.find((f) => f.label && (f.label.endsWith("::forward") || f.short_label === "forward"));
    if (forward) return fnId(forward);
    const entry = fns.find((f) => f.is_entrypoint && (f.operations || 0) <= 150);
    if (entry) return fnId(entry);
    return fns.length === 1 ? fnId(fns[0]) : "";
  }

  function populateFunctionFilter(data) {
    const sel = document.getElementById("fn-filter");
    const fns = data.functions || [];
    sel.innerHTML =
      '<option value="">— select function —</option>' +
      fns
        .map(
          (f) =>
            `<option value="${esc(fnId(f))}">${esc(f.short_label || f.label || fnId(f))}` +
            ` (${f.operations || 0} ops, ${f.tensors || 0} tensors)` +
            (f.is_entrypoint ? " *" : "") +
            "</option>"
        )
        .join("");
    if (!selectedFunction || !fns.some((f) => fnId(f) === selectedFunction))
      selectedFunction = defaultFunction(data);
    sel.value = selectedFunction;
  }

  function filterScopedDataflow(data) {
    if (!selectedFunction) return { nodes: [], edges: [], hint: true, total: (data.nodes || []).length };
    const fn = selectedFunction;
    const ops = (data.nodes || []).filter((n) => n.kind === "operation" && fnId(n.function) === fn);
    const opIds = new Set(ops.map((n) => fnId(n.id)));
    const linked = new Set();
    (data.edges || []).forEach((e) => {
      const from = fnId(e.from), to = fnId(e.to);
      if (opIds.has(to) || opIds.has(from)) { linked.add(from); linked.add(to); }
    });
    const nodes = (data.nodes || []).filter((n) => {
      if (n.kind === "operation") return fnId(n.function) === fn;
      return fnId(n.function) === fn || linked.has(fnId(n.id));
    });
    const ids = new Set(nodes.map((n) => fnId(n.id)));
    const edges = (data.edges || []).filter((e) => ids.has(fnId(e.from)) && ids.has(fnId(e.to)));
    return { nodes, edges, hint: false, total: (data.nodes || []).length };
  }

  function updateZoomLabel() {
    const el = document.getElementById("zoom-label");
    if (el) el.textContent = Math.round(graphView.k * 100) + "%";
  }

  function selectView(id, keepHighlights) {
    currentView = id;
    if (!keepHighlights) { highlights.clear(); selectedId = null; setInspector(null); }
    else if (selectedId) { selectedId = null; setInspector(null); }
    hoveredId = null;
    graphState = null;
    graphView = { x: 0, y: 0, k: 1, _fit: true };

    document.querySelectorAll("[data-view-tabs] .tab").forEach((t) => {
      t.setAttribute("aria-selected", t.dataset.view === id ? "true" : "false");
    });

    const meta = VIEW_META.find((v) => v.id === id) || VIEW_META[0];
    document.querySelector("[data-canvas-title]").textContent = meta.label;

    const treeSection = document.getElementById("tree-section");
    const searchSection = document.getElementById("search-section");
    if (treeSection) treeSection.hidden = id !== "architecture";
    if (searchSection) searchSection.hidden = id === "findings";

    const toolbar = document.getElementById("graph-toolbar");
    toolbar.hidden = !meta.scoped;
    if (meta.scoped) {
      populateFunctionFilter(viewData(id));
      document.getElementById("fn-filter").onchange = (e) => {
        selectedFunction = e.target.value;
        refreshGraph();
      };
    }

    document.getElementById("fit-btn").onclick = () => {
      if (graphState) { fitView(graphState.nodes); graphState.applyView(); updateZoomLabel(); }
      else { graphView._fit = true; refreshGraph(); }
    };
    document.getElementById("reset-btn").onclick = () => {
      graphView = { x: 0, y: 0, k: 1, _fit: true };
      refreshGraph();
    };

    if (meta.graph) refreshGraph();
    else clearCanvas("Select a finding and click \"Show in graph\" to highlight related nodes.");
    if (id === "architecture") buildTree();
    if (id === "findings") setInspector(null);
  }

  function clearCanvas(msg) {
    const svg = document.getElementById("graph-canvas");
    const empty = document.getElementById("empty-graph");
    const banner = document.getElementById("graph-banner");
    graphState = null;
    hoveredId = null;
    banner.hidden = true;
    document.querySelector("[data-graph-stats]").textContent = "";
    empty.hidden = false;
    empty.querySelector("p:last-child").textContent = msg || "No nodes to display.";
    svg.setAttribute("aria-hidden", "true");
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    hideTooltip();
    updateZoomLabel();
  }

  function refreshGraph() {
    const meta = VIEW_META.find((v) => v.id === currentView) || VIEW_META[0];
    if (!meta.graph) return;
    let data = viewData(currentView);
    const banner = document.getElementById("graph-banner");
    if (meta.scoped) {
      const filtered = filterScopedDataflow(data);
      if (filtered.hint) {
        banner.hidden = false;
        banner.textContent = `Full graph: ${filtered.total} nodes across ${(data.functions || []).length} functions — select a function to explore.`;
        clearCanvas("Select a function above to view its dataflow graph.");
        return;
      }
      banner.hidden = false;
      banner.textContent = `Showing ${filtered.nodes.length} of ${filtered.total} nodes`;
      data = filtered;
    } else {
      banner.hidden = true;
    }
    drawGraph(data, meta);
  }

  function buildTree() {
    const S = V.structure || {};
    const mods = S.modules || [];
    const params = S.parameters || [];
    const byParent = {}, byModule = {};
    mods.forEach((m) => { const p = m.parent == null ? "root" : String(m.parent); (byParent[p] = byParent[p] || []).push(m); });
    params.forEach((p) => { const mid = String(p.module_id || ""); (byModule[mid] = byModule[mid] || []).push(p); });
    const tree = document.getElementById("module-tree");
    const open = new Set(mods.map((m) => String(m.id)));

    function render() {
      const q = (document.getElementById("mod-search").value || "").toLowerCase();
      tree.innerHTML = "";
      function hit(m) {
        const label = (m.field ? m.field + ": " : "") + (m.type_name || "module");
        return !q || label.toLowerCase().includes(q) || String(m.prefix || "").toLowerCase().includes(q);
      }
      function add(items, depth) {
        items.forEach((m) => {
          const label = (m.field ? m.field + ": " : "") + (m.type_name || "module");
          const id = String(m.id);
          const ch = byParent[id] || [];
          const owned = byModule[id] || [];
          if (q && !hit(m) && !ch.some(hit)) return;
          const has = ch.length > 0 || owned.length > 0;
          const expanded = open.has(id) || !!q;
          const row = document.createElement("div");
          row.className = "tree-item";
          row.setAttribute("role", "treeitem");
          row.style.paddingLeft = depth * 14 + 4 + "px";
          if (has) {
            const b = document.createElement("button");
            b.type = "button"; b.className = "tw";
            b.textContent = expanded ? "▾" : "▸";
            b.onclick = (e) => { e.stopPropagation(); if (open.has(id)) open.delete(id); else open.add(id); render(); };
            row.appendChild(b);
          } else {
            const sp = document.createElement("span"); sp.className = "tw"; sp.textContent = "·"; row.appendChild(sp);
          }
          const t = document.createElement("span"); t.textContent = label; row.appendChild(t);
          const meta = document.createElement("span"); meta.className = "tn";
          meta.textContent = (owned.length ? owned.length + " params" : "") + (m.builder_root ? " · " + m.builder_root : "");
          row.appendChild(meta);
          row.onclick = () => {
            tree.querySelectorAll("[aria-selected=true]").forEach((x) => x.setAttribute("aria-selected", "false"));
            row.setAttribute("aria-selected", "true");
            setInspector(Object.assign({}, m, { label, kind: "module" }));
            highlightNode(id);
          };
          tree.appendChild(row);
          if (has && expanded) {
            owned.forEach((p) => {
              if (q && !String(p.key || "").toLowerCase().includes(q)) return;
              const pr = document.createElement("div");
              pr.className = "param-item"; pr.setAttribute("role", "treeitem");
              pr.style.paddingLeft = (depth + 1) * 14 + 18 + "px";
              pr.innerHTML = `<span class="pk">${esc(p.key || "param")}</span><span class="tn">${esc(p.dtype || "")}${p.grad_state ? " · " + esc(p.grad_state) : ""}</span>`;
              pr.onclick = () => {
                tree.querySelectorAll("[aria-selected=true]").forEach((x) => x.setAttribute("aria-selected", "false"));
                pr.setAttribute("aria-selected", "true");
                setInspector(Object.assign({}, p, { label: p.key, kind: "parameter" }));
                highlightNode(fnId(p.id));
              };
              tree.appendChild(pr);
            });
            add(ch, depth + 1);
          }
        });
      }
      add(byParent.root || mods.filter((m) => m.parent == null), 0);
    }
    document.getElementById("mod-search").oninput = render;
    render();
  }

  function highlightNode(id) {
    highlights = new Set([id]);
    const meta = VIEW_META.find((v) => v.id === currentView);
    if (meta && meta.graph && graphState) graphState.updateHighlight();
    else if (currentView !== "architecture") selectView("architecture", true);
  }

  function buildFindings() {
    const list = document.getElementById("findings-list");
    const items = (V.findings && V.findings.items) || [];
    list.innerHTML = items.length
      ? items
          .map((f) => {
            const cls = esc(f.severity || "Information");
            const view = f.suggested_view || "architecture";
            return `<div class="finding ${cls}" role="listitem" data-finding="${esc(idStr(f))}" data-view="${esc(view)}">
              <div class="sev">${cls}${f.rule ? " · " + esc(f.rule) : ""}</div>
              <div class="msg">${esc(f.message || "")}</div>
              <div class="goto">Show in ${esc(view.replace("_", " "))}</div>
            </div>`;
          })
          .join("")
      : '<div class="finding">No findings</div>';

    list.onclick = (e) => {
      const row = e.target.closest(".finding");
      if (!row || !row.dataset.finding) return;
      list.querySelectorAll(".finding.sel").forEach((x) => x.classList.remove("sel"));
      row.classList.add("sel");
      const f = items.find((x) => idStr(x) === row.dataset.finding);
      setInspector(Object.assign({}, f, { label: f.rule, kind: "finding" }));
      highlights = new Set((f.related || []).map(String));
      const targetView = f.suggested_view || row.dataset.view || "architecture";
      if (currentView !== targetView) selectView(targetView, true);
      else refreshGraph();
    };
  }

  function gradOf(n) {
    const g = n.grad_state || n.grad || "Unknown";
    return GRAD.includes(g) ? g : "Unknown";
  }

  function strokeFor(n) {
    if (n.kind === "component" || n.kind === "module") return "var(--module)";
    if (n.kind === "operation") return "var(--op)";
    if (n.kind === "parameter") return "var(--" + gradOf(n) + ")";
    if (n.kind === "stage") return "var(--stage)";
    return "var(--tensor)";
  }

  function svgPoint(svg, clientX, clientY) {
    const r = svg.getBoundingClientRect();
    return { x: clientX - r.left, y: clientY - r.top };
  }

  function fitView(vis) {
    if (!vis.length) return;
    const wrap = document.getElementById("canvas-wrap");
    const pw = wrap.clientWidth || 800;
    const ph = wrap.clientHeight || 400;
    const coords = vis.filter((n) => Number.isFinite(n._x) && Number.isFinite(n._y));
    if (!coords.length) return;
    const pad = 64;
    const minX = Math.min(...coords.map((n) => n._x));
    const maxX = Math.max(...coords.map((n) => n._x + (n._w || 0)));
    const minY = Math.min(...coords.map((n) => n._y));
    const maxY = Math.max(...coords.map((n) => n._y + (n._h || 0)));
    const gw = Math.max(maxX - minX + pad * 2, 1);
    const gh = Math.max(maxY - minY + pad * 2, 1);
    graphView.k = Math.min(2.5, Math.max(0.06, Math.min(pw / gw, ph / gh)));
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    graphView.x = pw / 2 - cx * graphView.k;
    graphView.y = ph / 2 - cy * graphView.k;
    updateZoomLabel();
  }

  function zoomAt(sx, sy, factor) {
    const k0 = graphView.k;
    const k1 = Math.min(4, Math.max(0.06, k0 * factor));
    const gx = (sx - graphView.x) / k0;
    const gy = (sy - graphView.y) / k0;
    graphView.k = k1;
    graphView.x = sx - gx * k1;
    graphView.y = sy - gy * k1;
    updateZoomLabel();
  }

  function neighborSet(nodeId, edges) {
    const s = new Set([nodeId]);
    edges.forEach((e) => { if (e._from === nodeId) s.add(e._to); if (e._to === nodeId) s.add(e._from); });
    return s;
  }

  function displayLabel(n) {
    return CGLayout.labelOf(n);
  }

  const tooltip = document.getElementById("graph-tooltip");

  function showTooltip(n, clientX, clientY) {
    if (!tooltip || !n) return;
    const wrap = document.getElementById("canvas-wrap");
    const r = wrap.getBoundingClientRect();
    tooltip.innerHTML =
      `<div class="tt-title">${esc(displayLabel(n))}</div>` +
      (n.qualified_name ? `<div class="tt-meta">${esc(n.qualified_name)}</div>` : "") +
      (n.shape ? `<div class="tt-meta">${esc(fmtShape(n.shape))}</div>` : "");
    tooltip.classList.add("visible");
    const tx = Math.min(clientX - r.left + 12, r.width - tooltip.offsetWidth - 8);
    const ty = Math.min(clientY - r.top + 12, r.height - tooltip.offsetHeight - 8);
    tooltip.style.left = Math.max(8, tx) + "px";
    tooltip.style.top = Math.max(8, ty) + "px";
  }

  function hideTooltip() {
    if (tooltip) tooltip.classList.remove("visible");
  }

  function buildNodeBody(n) {
    const kind = n.kind || "tensor";
    const kindLabel = KIND_LABEL[kind] || kind;
    const lines = n._titleLines || [displayLabel(n)];
    const sub = n._sub || CGLayout.sublabelOf(n);
    let html = `<div class="nb-head">`;
    html += `<span class="nb-kind">${esc(kindLabel)}</span>`;
    html += `</div>`;
    html += lines.map((l) => `<div class="nb-title">${esc(l)}</div>`).join("");
    if (sub) html += `<div class="nb-sub">${esc(sub)}</div>`;
    return html;
  }

  function drawGraph(data, meta) {
    const layout = meta.layout || "layered";
    const direction = meta.direction || "LR";
    const svg = document.getElementById("graph-canvas");
    const empty = document.getElementById("empty-graph");
    const NS = svg.namespaceURI;
    const XHTML = "http://www.w3.org/1999/xhtml";

    const nodes = (data.nodes || []).map((n, i) => Object.assign({}, n, { _id: fnId(n.id != null ? n.id : i) }));
    if (!nodes.length) { clearCanvas("No nodes match the current filter."); return; }
    empty.hidden = true;
    svg.removeAttribute("aria-hidden");

    const edges = (data.edges || []).map((e, i) =>
      Object.assign({}, e, { _id: String(e.id != null ? e.id : "e" + i), _from: fnId(e.from), _to: fnId(e.to) })
    );

    document.querySelector("[data-graph-stats]").textContent = nodes.length + " nodes · " + edges.length + " edges";

    if (layout === "tree") CGLayout.layoutTree(nodes, edges);
    else CGLayout.layoutLayered(nodes, edges, direction);

    const byId = {};
    nodes.forEach((n) => (byId[n._id] = n));
    const visEdges = edges.filter((e) => byId[e._from] && byId[e._to]);
    CGLayout.assignEdgePorts(nodes, visEdges, byId, layout, direction);

    let panDrag = null;
    const rootG = document.createElementNS(NS, "g");
    const bandG = document.createElementNS(NS, "g");
    const edgeG = document.createElementNS(NS, "g");
    const nodeG = document.createElementNS(NS, "g");
    rootG.appendChild(bandG);
    rootG.appendChild(edgeG);
    rootG.appendChild(nodeG);

    function focusSet() {
      const id = hoveredId || selectedId;
      if (!id) return null;
      return neighborSet(id, visEdges);
    }

    function updateClasses() {
      const focus = focusSet();
      nodeG.querySelectorAll(".node").forEach((el) => {
        const id = el.dataset.nodeId;
        el.classList.toggle("dim", focus && !focus.has(id) && id !== selectedId);
        el.classList.toggle("sel", id === selectedId);
        el.classList.toggle("highlight", highlights.has(id));
      });
      edgeG.querySelectorAll(".edge-group").forEach((el) => {
        const e = el.dataset;
        const lit = focus && (focus.has(e.from) || focus.has(e.to));
        const hi = highlights.has(e.from) || highlights.has(e.to);
        el.classList.toggle("dim", focus && !lit);
        el.classList.toggle("highlight", hi || lit);
        el.classList.toggle("sel", el.dataset.edgeId === selectedId);
      });
    }

    function buildSVG() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      rootG.setAttribute("transform", `translate(${graphView.x},${graphView.y}) scale(${graphView.k})`);
      svg.appendChild(rootG);
      bandG.innerHTML = "";
      edgeG.innerHTML = "";
      nodeG.innerHTML = "";

      CGLayout.layerBands(nodes, direction).forEach((b, i) => {
        const rect = document.createElementNS(NS, "rect");
        rect.setAttribute("class", "layer-band");
        rect.setAttribute("x", b.x);
        rect.setAttribute("y", b.y);
        rect.setAttribute("width", b.w);
        rect.setAttribute("height", b.h);
        rect.setAttribute("rx", "12");
        if (i % 2) rect.setAttribute("opacity", "0.55");
        bandG.appendChild(rect);
      });

      visEdges.forEach((e) => {
        const sev = (e.kind || "").includes("sever") || e.grad_state === "Severed";
        const edgeKind = CGLayout.edgeClass(e);
        const pathD = CGLayout.routeEdge(e);
        const g = document.createElementNS(NS, "g");
        g.setAttribute("class", "edge-group");
        g.dataset.from = e._from;
        g.dataset.to = e._to;
        g.dataset.edgeId = e._id;

        const hit = document.createElementNS(NS, "path");
        hit.setAttribute("class", "edge-hit");
        hit.setAttribute("d", pathD);
        hit.setAttribute("fill", "none");
        hit.setAttribute("stroke", "transparent");
        hit.setAttribute("stroke-width", "14");

        const p = document.createElementNS(NS, "path");
        p.setAttribute("class", "edge " + edgeKind);
        p.dataset.from = e._from;
        p.dataset.to = e._to;
        p.dataset.edgeId = e._id;
        p.setAttribute("stroke-width", sev ? "2.5" : "2");
        p.setAttribute("fill", "none");
        p.setAttribute("marker-end", sev ? "url(#arrow-sev)" : "url(#arrow)");
        p.setAttribute("d", pathD);

        const activate = (ev) => {
          ev.stopPropagation();
          selectedId = e._id;
          setInspector(Object.assign({ kind: "edge", label: e.label || e._id }, e));
          updateClasses();
        };
        hit.onclick = activate;
        p.onclick = activate;
        g.appendChild(hit);
        g.appendChild(p);
        edgeG.appendChild(g);
      });

      nodes.forEach((n) => {
        const gg = document.createElementNS(NS, "g");
        const kind = n.kind || "tensor";
        gg.setAttribute("class", "node " + kind);
        gg.dataset.nodeId = n._id;
        gg.setAttribute("tabindex", "0");
        gg.setAttribute("role", "button");

        const card = document.createElementNS(NS, "rect");
        card.setAttribute("class", "node-card");
        card.setAttribute("x", n._x);
        card.setAttribute("y", n._y);
        card.setAttribute("width", n._w);
        card.setAttribute("height", n._h);
        card.setAttribute("rx", kind === "tensor" || kind === "parameter" ? "10" : "8");
        card.setAttribute("stroke", strokeFor(n));
        gg.appendChild(card);

        const fo = document.createElementNS(NS, "foreignObject");
        fo.setAttribute("x", n._x);
        fo.setAttribute("y", n._y);
        fo.setAttribute("width", n._w);
        fo.setAttribute("height", n._h);
        const div = document.createElementNS(XHTML, "div");
        div.setAttribute("class", "node-body");
        div.innerHTML = buildNodeBody(n);
        fo.appendChild(div);
        gg.appendChild(fo);

        gg.onmouseenter = (ev) => {
          hoveredId = n._id;
          updateClasses();
          showTooltip(n, ev.clientX, ev.clientY);
        };
        gg.onmouseleave = () => {
          if (hoveredId === n._id) hoveredId = null;
          updateClasses();
          hideTooltip();
        };
        gg.onmousemove = (ev) => showTooltip(n, ev.clientX, ev.clientY);
        gg.onclick = (ev) => {
          ev.stopPropagation();
          selectedId = n._id;
          setInspector(n);
          updateClasses();
        };
        nodeG.appendChild(gg);
      });

      updateClasses();
    }

    function ensureDefs() {
      let defs = svg.querySelector("defs");
      if (!defs) {
        defs = document.createElementNS(NS, "defs");
        svg.insertBefore(defs, svg.firstChild);
      }
      if (!svg.querySelector("#arrow")) {
        [["arrow", "var(--edge)"], ["arrow-sev", "var(--edge-severed)"]].forEach(([id, color]) => {
          const m = document.createElementNS(NS, "marker");
          m.setAttribute("id", id);
          m.setAttribute("viewBox", "0 0 10 10");
          m.setAttribute("refX", "9"); m.setAttribute("refY", "5");
          m.setAttribute("markerWidth", "8"); m.setAttribute("markerHeight", "8");
          m.setAttribute("orient", "auto-start-reverse");
          const path = document.createElementNS(NS, "path");
          path.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
          path.setAttribute("fill", color);
          m.appendChild(path);
          defs.appendChild(m);
        });
      }
    }

    function applyView() {
      rootG.setAttribute("transform", `translate(${graphView.x},${graphView.y}) scale(${graphView.k})`);
    }

    ensureDefs();
    buildSVG();

    if (graphView._fit || (graphView.k === 1 && graphView.x === 0 && graphView.y === 0)) {
      const runFit = () => { fitView(nodes); applyView(); };
      runFit();
      requestAnimationFrame(runFit);
      graphView._fit = false;
    }

    svg.onpointerdown = (e) => {
      if (isGraphItemTarget(e.target)) return;
      hideTooltip();
      const p = svgPoint(svg, e.clientX, e.clientY);
      panDrag = { px: p.x, py: p.y, vx: graphView.x, vy: graphView.y, moved: false, sx: e.clientX, sy: e.clientY };
      svg.setPointerCapture(e.pointerId);
    };
    svg.onpointermove = (e) => {
      if (!panDrag) return;
      if (!panDrag.moved) {
        const dx = e.clientX - panDrag.sx;
        const dy = e.clientY - panDrag.sy;
        if (dx * dx + dy * dy > 16) panDrag.moved = true;
      }
      if (!panDrag.moved) return;
      const p = svgPoint(svg, e.clientX, e.clientY);
      graphView.x = panDrag.vx + (p.x - panDrag.px);
      graphView.y = panDrag.vy + (p.y - panDrag.py);
      applyView();
    };
    svg.onpointerup = (e) => {
      if (panDrag && !panDrag.moved && !isGraphItemTarget(e.target)) clearSelection(false);
      panDrag = null;
    };
    svg.onpointercancel = () => { panDrag = null; };
    svg.onwheel = (e) => {
      e.preventDefault();
      const p = svgPoint(svg, e.clientX, e.clientY);
      zoomAt(p.x, p.y, e.deltaY < 0 ? 1.12 : 0.89);
      applyView();
    };

    graphState = { nodes, edges, layout, applyView, updateHighlight: updateClasses, rebuild: buildSVG };
    updateZoomLabel();
  }

  function initNodeSearch() {
    const input = document.getElementById("node-search");
    if (!input) return;
    input.addEventListener("input", () => {
      const q = input.value.toLowerCase();
      if (!q || !graphState) return;
      const match = graphState.nodes.find((n) => {
        const labels = [n.label, n.short_label, n.key, n._id, n.qualified_name].filter(Boolean).map(String);
        return labels.some((l) => l.toLowerCase().includes(q));
      });
      if (match) {
        selectedId = match._id;
        setInspector(match);
        graphState.updateHighlight();
        if (Number.isFinite(match._cx) && Number.isFinite(match._cy)) {
          const wrap = document.getElementById("canvas-wrap");
          graphView.x = wrap.clientWidth / 2 - match._cx * graphView.k;
          graphView.y = wrap.clientHeight / 2 - match._cy * graphView.k;
          graphState.applyView();
        }
      }
    });
  }

  function initCanvasControls() {
    const wrap = document.getElementById("canvas-wrap");
    document.getElementById("zoom-in").onclick = () => {
      if (!graphState) return;
      zoomAt(wrap.clientWidth / 2, wrap.clientHeight / 2, 1.2);
      graphState.applyView();
    };
    document.getElementById("zoom-out").onclick = () => {
      if (!graphState) return;
      zoomAt(wrap.clientWidth / 2, wrap.clientHeight / 2, 1 / 1.2);
      graphState.applyView();
    };
    document.getElementById("zoom-fit").onclick = () => {
      if (!graphState) return;
      fitView(graphState.nodes);
      graphState.applyView();
    };
  }

  function initLegendToggle() {
    const box = document.querySelector(".legend-float");
    const btn = document.getElementById("legend-toggle");
    if (!box || !btn) return;
    btn.onclick = () => box.classList.toggle("collapsed");
  }

  function exportSVG() {
    const svg = document.getElementById("graph-canvas");
    if (!svg.firstChild) return;
    const clone = svg.cloneNode(true);
    clone.setAttribute("xmlns", "http" + "://www.w3.org/2000/svg");
    clone.setAttribute("width", "100%");
    clone.setAttribute("height", "100%");
    const title = document.createElementNS("http" + "://www.w3.org/2000/svg", "title");
    title.textContent = (P.package || "candle-graph") + " — " + currentView;
    clone.insertBefore(title, clone.firstChild);

    const style = document.createElementNS("http" + "://www.w3.org/2000/svg", "style");
    style.textContent = `
      .node-card { fill: #fff; stroke-width: 1.5; }
      .node-body { font-family: system-ui, sans-serif; font-size: 12px; color: #0f172a; }
      .edge-default, .edge-in { stroke: #64748b; }
      .edge-out { stroke: #2563eb; }
      .edge-severed { stroke: #dc2626; }
      .edge { fill: none; stroke-width: 2; }
      .edge-hit { display: none; }
    `;
    clone.insertBefore(style, clone.firstChild);

    const blob = new Blob(
      ['<?xml version="1.0" encoding="UTF-8"?>\n', new XMLSerializer().serializeToString(clone)],
      { type: "image/svg+xml" }
    );
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = (P.package || "model") + "-" + currentView + ".svg";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function toggleExportMode() {
    const on = root.getAttribute("data-export") === "true";
    root.setAttribute("data-export", on ? "false" : "true");
    if (!on && graphState) {
      graphView._fit = true;
      fitView(graphState.nodes);
      graphState.applyView();
    }
  }

  function initKeyboard() {
    document.addEventListener("keydown", (e) => {
      if (e.target.matches("input, select, textarea")) return;
      if (e.key === "u" || e.key === "U" || e.key === "Escape") {
        if (selectedId || hoveredId) {
          e.preventDefault();
          clearSelection(false);
        }
        return;
      }
      if (!graphState) return;
      const wrap = document.getElementById("canvas-wrap");
      const cx = wrap.clientWidth / 2;
      const cy = wrap.clientHeight / 2;
      if (e.key === "+" || e.key === "=") {
        zoomAt(cx, cy, 1.2);
        graphState.applyView();
      } else if (e.key === "-") {
        zoomAt(cx, cy, 1 / 1.2);
        graphState.applyView();
      } else if (e.key === "f" || e.key === "F") {
        fitView(graphState.nodes);
        graphState.applyView();
      } else if (e.key === "0") {
        graphView = { x: 0, y: 0, k: 1, _fit: true };
        refreshGraph();
      }
    });
  }

  document.getElementById("theme-btn").onclick = () => {
    const n = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", n);
    localStorage.setItem("cg-theme", n);
  };
  document.getElementById("export-btn").onclick = exportSVG;
  document.getElementById("print-btn").onclick = toggleExportMode;

  initResizers();
  initTabs();
  initCanvasControls();
  initLegendToggle();
  initKeyboard();
  renderCoverage();
  buildTree();
  buildFindings();
  initNodeSearch();
  setInspector(null);
  selectView("architecture");
})();
