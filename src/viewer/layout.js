/* candle-graph layout — powered by dagre (MIT, @dagrejs/dagre 3.1) */
window.CGLayout = (function () {
  "use strict";

  var dagreLib = typeof dagre !== "undefined" ? dagre : (typeof window !== "undefined" ? window.dagre : null);

  function prettyTensorLabel(name) {
    name = String(name || "");
    if (name.indexOf("result_of_") === 0) name = name.slice(10);
    if (name.slice(-9) === "::forward") name = name.slice(0, -9);
    var dot = name.lastIndexOf(".");
    if (dot >= 0) name = name.slice(dot + 1);
    return name.replace(/_/g, " ").trim() || "tensor";
  }

  function labelOf(n) {
    var kind = n.kind || "tensor";
    if (n.short_label) return String(n.short_label);
    var raw = n.label || n.key || n._id || "";
    if (kind === "tensor" || kind === "parameter") return prettyTensorLabel(raw);
    return raw;
  }

  function sublabelOf(n) {
    if (n.kind === "operation") {
      if (n.timing && n.timing.avg_ns != null) {
        var t = n.timing;
        var avg = (t.avg_ns / 1e6).toFixed(2);
        var lo = (t.min_ns / 1e6).toFixed(2);
        var hi = (t.max_ns / 1e6).toFixed(2);
        return avg + "ms avg · " + lo + "–" + hi + "ms (" + t.samples + "×)";
      }
      if (n.gradient_rule) return n.gradient_rule;
      return "";
    }
    if (n.kind === "parameter" || n.kind === "tensor") {
      var g = n.grad_state || n.grad || "";
      if (g === "Unknown") g = "";
      var d = n.dtype && n.dtype !== "Unknown" ? n.dtype : "";
      if (!g && !d) return "";
      if (g && d) return g + " · " + d;
      return g || d;
    }
    if (n.kind === "component" || n.kind === "module") {
      var parts = [];
      if (n.parameters != null) parts.push(n.parameters + " params");
      if (n.builder_root) parts.push(n.builder_root);
      return parts.join(" · ");
    }
    if (n.shape) {
      return Array.isArray(n.shape) ? n.shape.join(" × ") : String(n.shape);
    }
    return "";
  }

  function textWidth(text, charW) {
    return Math.max(0, String(text || "").length) * charW;
  }

  function wrapLines(text, maxChars) {
    text = String(text || "");
    if (text.length <= maxChars) return [text];
    var words = text.split(/[\s._:/\\-]+/);
    var lines = [];
    var cur = "";
    words.forEach(function (w) {
      if (!w) return;
      var next = cur ? cur + " " + w : w;
      if (next.length > maxChars && cur) {
        lines.push(cur);
        cur = w;
      } else {
        cur = next;
      }
    });
    if (cur) lines.push(cur);
    return lines.length ? lines.slice(0, 3) : [text.slice(0, maxChars)];
  }

  function nodeDims(n, layout) {
    var label = labelOf(n);
    var sub = sublabelOf(n);
    var kind = n.kind || "tensor";
    var padX = 24;
    var padY = 12;
    var lineH = 15;
    var subH = sub ? 13 : 0;
    var kindBadgeH = 14;
    var minW, maxW, maxChars;

    if (layout === "tree") {
      minW = kind === "component" ? 200 : 180;
      maxW = 280;
      maxChars = 28;
    } else if (kind === "operation") {
      padX = 16;
      padY = 10;
      kindBadgeH = 16;
      minW = 140;
      maxW = 220;
      maxChars = 22;
    } else if (kind === "parameter") {
      minW = 132;
      maxW = 228;
      maxChars = 28;
    } else if (kind === "stage") {
      padX = 16;
      padY = 10;
      kindBadgeH = 16;
      minW = 130;
      maxW = 200;
      maxChars = 22;
    } else if (kind === "component" || kind === "module") {
      padX = 16;
      padY = 10;
      kindBadgeH = 16;
      minW = 180;
      maxW = 260;
      maxChars = 26;
    } else {
      minW = 132;
      maxW = 240;
      maxChars = 30;
    }

    var titleLines = wrapLines(label, maxChars);
    var titleW = Math.max.apply(null, titleLines.map(function (l) { return textWidth(l, 7.4); }));
    var subW = sub ? textWidth(sub, 6.4) : 0;
    var kindW = kind === "tensor" || kind === "parameter" ? textWidth(kind, 6) + 16 : 0;
    var w = Math.min(maxW, Math.max(minW, Math.max(titleW, subW, kindW) + padX * 2));
    var h = padY * 2 + kindBadgeH + titleLines.length * lineH + subH + (sub ? 4 : 0);

    n._titleLines = titleLines;
    n._sub = sub;
    return { w: Math.ceil(w), h: Math.ceil(h) };
  }

  function pointsToPath(points) {
    if (!points || points.length < 2) return "";
    var d = "M" + points[0].x + " " + points[0].y;
    for (var i = 1; i < points.length; i++) {
      d += " L" + points[i].x + " " + points[i].y;
    }
    return d;
  }

  function edgeKey(e) {
    return { v: e._from, w: e._to, name: e._key };
  }

  function fallbackLayout(nodes, direction) {
    var LR = direction !== "TB";
    var gapX = LR ? 260 : 48;
    var gapY = LR ? 52 : 100;
    nodes.forEach(function (n, i) {
      var sz = nodeDims(n, "layered");
      n._w = sz.w;
      n._h = sz.h;
      if (LR) {
        n._x = (i % 8) * gapX;
        n._y = Math.floor(i / 8) * gapY;
      } else {
        n._x = (i % 6) * gapX;
        n._y = Math.floor(i / 6) * gapY;
      }
      n._cx = n._x + n._w / 2;
      n._cy = n._y + n._h / 2;
    });
  }

  function layoutWithDagre(nodes, edges, opts) {
    if (!dagreLib || !dagreLib.graphlib || !dagreLib.layout) {
      fallbackLayout(nodes, opts.direction || "LR");
      return;
    }

    var g = new dagreLib.graphlib.Graph({ multigraph: true, compound: false });
    var rankdir = opts.direction === "TB" ? "TB" : "LR";
    var layoutKind = opts.layout || "layered";

    g.setGraph({
      rankdir: rankdir,
      ranker: "network-simplex",
      acyclicer: "greedy",
      align: "UL",
      nodesep: opts.nodesep || 80,
      edgesep: opts.edgesep || 64,
      ranksep: opts.ranksep || 200,
      marginx: 56,
      marginy: 56,
    });
    g.setDefaultEdgeLabel(function () { return {}; });

    nodes.forEach(function (n) {
      var sz = nodeDims(n, layoutKind);
      g.setNode(n._id, { width: sz.w, height: sz.h });
    });

    edges.forEach(function (e, i) {
      if (!g.hasNode(e._from) || !g.hasNode(e._to)) return;
      e._key = "e" + i;
      g.setEdge(
        { v: e._from, w: e._to, name: e._key },
        { minlen: 1, weight: 1 }
      );
    });

    try {
      dagreLib.layout(g);
    } catch (err) {
      console.error("dagre layout failed:", err);
      fallbackLayout(nodes, opts.direction || "LR");
      return;
    }

    nodes.forEach(function (n) {
      var d = g.node(n._id);
      if (!d) return;
      n._w = d.width;
      n._h = d.height;
      n._x = d.x - d.width / 2;
      n._y = d.y - d.height / 2;
      n._cx = d.x;
      n._cy = d.y;
      n._col = d.rank;
      n._layer = d.rank;
    });

    edges.forEach(function (e) {
      if (!e._key) return;
      var edgeData = g.edge(edgeKey(e));
      e._points = edgeData && edgeData.points ? edgeData.points : null;
    });
  }

  function layoutLayered(nodes, edges, direction) {
    layoutWithDagre(nodes, edges, {
      layout: "layered",
      direction: direction || "LR",
      nodesep: 76,
      edgesep: 68,
      ranksep: 210,
    });
  }

  function layoutTree(nodes, edges) {
    layoutWithDagre(nodes, edges, {
      layout: "tree",
      direction: "LR",
      nodesep: 64,
      edgesep: 56,
      ranksep: 220,
    });
  }

  function layerBands(nodes, direction) {
    var LR = direction !== "TB";
    var byLayer = {};
    nodes.forEach(function (n) {
      if (!Number.isFinite(n._layer)) return;
      var L = n._layer;
      if (!byLayer[L]) {
        byLayer[L] = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity, layer: L };
      }
      var b = byLayer[L];
      b.minX = Math.min(b.minX, n._x);
      b.maxX = Math.max(b.maxX, n._x + n._w);
      b.minY = Math.min(b.minY, n._y);
      b.maxY = Math.max(b.maxY, n._y + n._h);
    });
    var pad = 20;
    return Object.keys(byLayer).map(function (k) {
      var b = byLayer[k];
      return {
        x: b.minX - pad,
        y: b.minY - pad,
        w: b.maxX - b.minX + pad * 2,
        h: b.maxY - b.minY + pad * 2,
      };
    });
  }

  /** Anchor on node boundary (not center) for cleaner orthogonal routing. */
  function portPoint(node, side, along) {
    var pad = 4;
    var cx = node._cx;
    var cy = node._cy;
    if (along != null) {
      if (side === "east" || side === "west") cy = along;
      else cx = along;
    }
    switch (side) {
      case "east":
        return { x: node._x + node._w + pad, y: cy };
      case "west":
        return { x: node._x - pad, y: cy };
      case "south":
        return { x: cx, y: node._y + node._h + pad };
      case "north":
        return { x: cx, y: node._y - pad };
      default:
        return { x: node._cx, y: node._cy };
    }
  }

  /** Spread attachment points along a node face so parallel edges do not overlap. */
  function spreadPort(node, side, index, total) {
    var margin = 10;
    var span = side === "east" || side === "west" ? node._h : node._w;
    var usable = Math.max(span - margin * 2, 8);
    var t = (index + 1) / (total + 1);
    var along = (side === "east" || side === "west" ? node._y : node._x) + margin + t * usable;
    return portPoint(node, side, along);
  }

  function choosePortSides(from, to, direction) {
    var LR = direction !== "TB";
    if (LR) {
      if (from._layer != null && to._layer != null && from._layer === to._layer) {
        var dy = to._cy - from._cy;
        if (dy >= 0) return { from: "south", to: "north" };
        return { from: "north", to: "south" };
      }
      return { from: "east", to: "west" };
    }
    if (from._layer != null && to._layer != null && from._layer === to._layer) {
      var dx = to._cx - from._cx;
      if (dx >= 0) return { from: "east", to: "west" };
      return { from: "west", to: "east" };
    }
    return { from: "south", to: "north" };
  }

  function orthogonalRoute(fromPt, toPt, direction, laneOffset) {
    var LR = direction !== "TB";
    var offset = laneOffset || 0;
    var pts = [fromPt];
    if (LR) {
      var midX = (fromPt.x + toPt.x) / 2 + offset;
      if (Math.abs(fromPt.y - toPt.y) < 2) {
        pts.push({ x: toPt.x, y: toPt.y });
      } else {
        pts.push({ x: midX, y: fromPt.y });
        pts.push({ x: midX, y: toPt.y });
        pts.push({ x: toPt.x, y: toPt.y });
      }
    } else {
      var midY = (fromPt.y + toPt.y) / 2 + offset;
      if (Math.abs(fromPt.x - toPt.x) < 2) {
        pts.push({ x: toPt.x, y: toPt.y });
      } else {
        pts.push({ x: fromPt.x, y: midY });
        pts.push({ x: toPt.x, y: midY });
        pts.push({ x: toPt.x, y: toPt.y });
      }
    }
    return pts;
  }

  function laneKeyForEdge(from, to, direction) {
    var LR = direction !== "TB";
    if (LR) {
      var lo = Math.min(from._layer != null ? from._layer : 0, to._layer != null ? to._layer : 0);
      var hi = Math.max(from._layer != null ? from._layer : 0, to._layer != null ? to._layer : 0);
      return "L" + lo + "-" + hi + ":" + Math.round(from._cy) + ">" + Math.round(to._cy);
    }
    var loX = Math.min(from._layer != null ? from._layer : 0, to._layer != null ? to._layer : 0);
    var hiX = Math.max(from._layer != null ? from._layer : 0, to._layer != null ? to._layer : 0);
    return "T" + loX + "-" + hiX + ":" + Math.round(from._cx) + ">" + Math.round(to._cx);
  }

  function assignEdgePorts(nodes, edges, byId, layout, direction) {
    var dir = direction || "LR";
    var fromGroups = {};
    var toGroups = {};
    var laneGroups = {};

    edges.forEach(function (e, i) {
      var from = byId[e._from];
      var to = byId[e._to];
      if (!from || !to) return;
      var sides = choosePortSides(from, to, dir);
      e._fromSide = sides.from;
      e._toSide = sides.to;
      var fk = e._from + ":" + sides.from;
      var tk = e._to + ":" + sides.to;
      if (!fromGroups[fk]) fromGroups[fk] = [];
      if (!toGroups[tk]) toGroups[tk] = [];
      fromGroups[fk].push(i);
      toGroups[tk].push(i);
    });

    edges.forEach(function (e, i) {
      var from = byId[e._from];
      var to = byId[e._to];
      if (!from || !to || !e._fromSide) return;

      if (e._points && e._points.length >= 2) {
        e._path = pointsToPath(e._points);
        return;
      }

      var fk = e._from + ":" + e._fromSide;
      var tk = e._to + ":" + e._toSide;
      var fi = fromGroups[fk].indexOf(i);
      var ft = fromGroups[fk].length;
      var ti = toGroups[tk].indexOf(i);
      var tt = toGroups[tk].length;

      e._fromPort = spreadPort(from, e._fromSide, fi, ft);
      e._toPort = spreadPort(to, e._toSide, ti, tt);

      var laneKey = laneKeyForEdge(from, to, dir);
      if (!laneGroups[laneKey]) laneGroups[laneKey] = [];
      laneGroups[laneKey].push(i);
    });

    Object.keys(laneGroups).forEach(function (key) {
      var group = laneGroups[key];
      var laneStep = 16;
      group.forEach(function (edgeIdx, laneIdx) {
        var e = edges[edgeIdx];
        var offset = (laneIdx - (group.length - 1) / 2) * laneStep;
        e._path = pointsToPath(orthogonalRoute(e._fromPort, e._toPort, dir, offset));
      });
    });
  }

  function routeEdge(e) {
    if (e._path) return e._path;
    if (e._points && e._points.length >= 2) return pointsToPath(e._points);
    return "";
  }

  function edgeMidpoint(e) {
    var pts = e._points;
    if (!pts || pts.length === 0) return null;
    var p = pts[Math.floor(pts.length / 2)];
    return { x: p.x, y: p.y - 4 };
  }

  function edgeClass(e) {
    if ((e.kind || "").indexOf("sever") >= 0 || e.grad_state === "Severed") return "edge-severed";
    if (e.kind === "call") return "edge-call";
    if (e.label && e.label.indexOf("ms") >= 0) return "edge-timed";
    if (e.label === "in") return "edge-in";
    if (e.label === "out") return "edge-out";
    if (e.kind === "composition") return "edge-composition";
    if (e.kind === "depends_on" || e.kind === "sequence") return "edge-pipeline";
    return "edge-default";
  }

  return {
    nodeDims: nodeDims,
    labelOf: labelOf,
    sublabelOf: sublabelOf,
    layoutLayered: layoutLayered,
    layoutTree: layoutTree,
    layerBands: layerBands,
    assignEdgePorts: assignEdgePorts,
    routeEdge: routeEdge,
    edgeMidpoint: edgeMidpoint,
    edgeClass: edgeClass,
  };
})();
