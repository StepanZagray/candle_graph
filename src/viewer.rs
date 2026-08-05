//! Dependency-free interactive HTML visualizer for `candle-graph/viewer/3` trace payloads.
//!
//! Embeds escaped JSON, CSS, and JS into one document. No CDN or network fetches.

use serde_json::Value;

use crate::graph::ExecutionGraph;

pub mod trace_view;

const CSS: &str = include_str!("viewer/style.css");
const TRACE_JS: &str = include_str!("viewer/app_trace.js");

/// Render a trace-only HTML document from an [`ExecutionGraph`].
pub fn render_trace_html(graph: &ExecutionGraph) -> String {
    let projection = trace_view::project(graph);
    render_trace_document(&projection)
}

fn render_trace_document(projection: &Value) -> String {
    let payload = embed_json(projection);
    let mut html = String::with_capacity(8192 + CSS.len() + TRACE_JS.len() + payload.len());
    html.push_str("<!DOCTYPE html>\n<html lang=\"en\" data-viewer=\"candle-graph-trace\">\n<head>\n");
    html.push_str("<meta charset=\"utf-8\"/>\n");
    html.push_str("<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>\n");
    html.push_str("<title>candle-graph trace</title>\n<style>");
    html.push_str(CSS);
    html.push_str(
        "\n.span-tree .span-row{display:flex;align-items:baseline;gap:6px;padding:4px 8px;border-radius:var(--radius-sm);cursor:pointer;font-size:12px;line-height:1.35}\n\
.span-tree .span-row:hover,.span-tree .span-row[aria-selected=true]{background:var(--accent-soft)}\n\
.span-tree .span-name{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:var(--font-mono);font-size:11px}\n\
.span-tree .span-ms{color:var(--muted);font-variant-numeric:tabular-nums;font-size:11px;white-space:nowrap}\n\
.span-tree .tw{width:14px;flex-shrink:0;color:var(--muted);border:none;background:none;padding:0;cursor:pointer;font-size:11px}\n\
.timeline-table{width:100%;border-collapse:collapse;font-size:12px}\n\
.timeline-table th,.timeline-table td{padding:6px 10px;border-bottom:1px solid var(--border);text-align:left}\n\
.memory-chart{width:100%;height:220px;display:block;background:var(--surface-2);border-radius:var(--radius-sm);margin-bottom:8px}\n\
.peak-table{font-size:11px}\n\
.peak-table table{width:100%;border-collapse:collapse}\n\
.peak-table th,.peak-table td{padding:4px 8px;border-bottom:1px solid var(--border);text-align:left;font-family:var(--font-mono)}\n\
.timeline-table tbody tr{cursor:pointer}\n\
.timeline-table tbody tr:hover,.timeline-table tbody tr.sel{background:var(--accent-soft)}\n\
.node.function .node-card,.node.root .node-card,.node.module .node-card{stroke-width:2.5}\n",
    );
    html.push_str("</style>\n</head>\n<body>\n");
    html.push_str("<a class=\"skip\" href=\"#canvas-pane\">Skip to canvas</a>\n");

    html.push_str("<header class=\"top\" role=\"banner\">\n");
    html.push_str("  <div class=\"brand\"><span class=\"brand-mark\" aria-hidden=\"true\"></span>candle-graph trace</div>\n");
    html.push_str("  <div class=\"cov\" data-coverage role=\"status\" aria-live=\"polite\"></div>\n");
    html.push_str("  <div class=\"top-actions\">\n");
    html.push_str("    <button type=\"button\" class=\"btn primary\" id=\"export-btn\" aria-label=\"Export graph as SVG\">Export SVG</button>\n");
    html.push_str("    <button type=\"button\" class=\"btn\" id=\"theme-btn\" data-theme-toggle aria-label=\"Toggle color theme\">Theme</button>\n");
    html.push_str("  </div>\n");
    html.push_str("</header>\n");

    html.push_str("<div class=\"layout\" id=\"app\">\n");
    html.push_str("  <nav class=\"pane pane-sidebar\" data-pane=\"sidebar\" aria-label=\"Navigation\">\n");
    html.push_str("    <div class=\"pane-h\">Views</div>\n");
    html.push_str("    <div class=\"tabs\" data-view-tabs role=\"tablist\" aria-label=\"Trace views\"></div>\n");
    html.push_str("    <div class=\"pane-h sub\">Span hierarchy</div>\n");
    html.push_str("    <div class=\"search-box\">\n");
    html.push_str("      <label class=\"sr\" for=\"span-search\">Search spans</label>\n");
    html.push_str("      <input id=\"span-search\" data-span-search type=\"search\" placeholder=\"Filter spans…\" autocomplete=\"off\"/>\n");
    html.push_str("    </div>\n");
    html.push_str("    <div id=\"span-tree\" class=\"scroll-area span-tree\" data-span-tree tabindex=\"0\" role=\"tree\" aria-label=\"Span hierarchy\"></div>\n");
    html.push_str("  </nav>\n");
    html.push_str("  <div class=\"resize-handle\" data-side=\"left\" aria-hidden=\"true\"></div>\n");

    html.push_str("  <main class=\"pane pane-canvas\" data-pane=\"canvas\" id=\"canvas-pane\" aria-label=\"Trace canvas\">\n");
    html.push_str("    <div class=\"pane-h row\"><span data-canvas-title>Trace</span><span class=\"graph-stats\" data-graph-stats></span></div>\n");
    html.push_str("    <div class=\"graph-toolbar\" id=\"graph-toolbar\">\n");
    html.push_str("      <button type=\"button\" class=\"btn\" id=\"fit-btn\" aria-label=\"Fit graph to view\">Fit</button>\n");
    html.push_str("      <button type=\"button\" class=\"btn\" id=\"reset-btn\" aria-label=\"Reset zoom and pan\">Reset</button>\n");
    html.push_str("    </div>\n");
    html.push_str("    <div class=\"canvas-wrap\" id=\"canvas-wrap\">\n");
    html.push_str("      <div id=\"empty-graph\" class=\"empty\" data-empty-state hidden>\n");
    html.push_str("        <p><strong>No trace data</strong></p>\n");
    html.push_str("        <p>This trace has no spans to display.</p>\n");
    html.push_str("      </div>\n");
    html.push_str("      <div id=\"timeline-panel\" class=\"scroll-area\" hidden></div>\n");
    html.push_str("      <svg id=\"graph-canvas\" data-canvas role=\"img\" aria-label=\"Trace graph\" tabindex=\"0\"></svg>\n");
    html.push_str("      <div id=\"graph-tooltip\" class=\"graph-tooltip\" role=\"tooltip\" aria-hidden=\"true\"></div>\n");
    html.push_str("      <div class=\"legend-float\" data-legend role=\"list\" aria-label=\"Heat legend\">\n");
    html.push_str("        <button type=\"button\" class=\"legend-toggle\" id=\"legend-toggle\" aria-expanded=\"true\">Legend</button>\n");
    html.push_str("        <div class=\"legend-body\">\n");
    html.push_str("          <span role=\"listitem\"><label><input type=\"radio\" name=\"heat-mode\" value=\"time\" checked> Self time heat</label></span>\n");
    html.push_str("          <span role=\"listitem\"><label><input type=\"radio\" name=\"heat-mode\" value=\"memory\"> Memory heat</label></span>\n");
    html.push_str("          <span role=\"listitem\" class=\"lg kind-op\">Call edge (ms label)</span>\n");
    html.push_str("          <span role=\"listitem\" class=\"lg kind-module\">Data edge (ms label)</span>\n");
    html.push_str("        </div>\n");
    html.push_str("      </div>\n");
    html.push_str("      <div class=\"canvas-controls\" aria-label=\"Zoom controls\">\n");
    html.push_str("        <button type=\"button\" class=\"btn icon\" id=\"zoom-in\" aria-label=\"Zoom in\">+</button>\n");
    html.push_str("        <div class=\"zoom-label\" id=\"zoom-label\">100%</div>\n");
    html.push_str("        <button type=\"button\" class=\"btn icon\" id=\"zoom-out\" aria-label=\"Zoom out\">−</button>\n");
    html.push_str("        <button type=\"button\" class=\"btn icon\" id=\"zoom-fit\" aria-label=\"Fit to view\">⤢</button>\n");
    html.push_str("      </div>\n");
    html.push_str("    </div>\n");
    html.push_str("  </main>\n");

    html.push_str("  <div class=\"resize-handle\" data-side=\"right\" aria-hidden=\"true\"></div>\n");
    html.push_str("  <aside class=\"pane pane-inspector\" data-pane=\"inspector\" aria-label=\"Inspector\">\n");
    html.push_str("    <div class=\"pane-h\">Inspector</div>\n");
    html.push_str("    <dl id=\"inspector\" data-inspector>\n");
    html.push_str("      <div><dt>Label</dt><dd data-field=\"label\">—</dd></div>\n");
    html.push_str("      <div><dt>Kind</dt><dd data-field=\"kind\">—</dd></div>\n");
    html.push_str("      <div><dt>Self time</dt><dd data-field=\"self_time\">—</dd></div>\n");
    html.push_str("      <div><dt>Total time</dt><dd data-field=\"total_time\">—</dd></div>\n");
    html.push_str("      <div><dt>Shape</dt><dd data-field=\"shape\">—</dd></div>\n");
    html.push_str("      <div><dt>Dtype</dt><dd data-field=\"dtype\">—</dd></div>\n");
    html.push_str("      <div><dt>Storage</dt><dd data-field=\"storage\">—</dd></div>\n");
    html.push_str("      <div><dt>Peak live</dt><dd data-field=\"peak_bytes\">—</dd></div>\n");
    html.push_str("      <div><dt>Requested</dt><dd data-field=\"bytes\">—</dd></div>\n");
    html.push_str("    </dl>\n");
    html.push_str("    <div class=\"pane-h sub\">Peak breakdown</div>\n");
    html.push_str("    <div id=\"peak-breakdown\" class=\"scroll-area peak-table\" data-peak-breakdown tabindex=\"0\"></div>\n");
    html.push_str("  </aside>\n");
    html.push_str("</div>\n");

    html.push_str("<script id=\"cg-payload\" type=\"application/json\">");
    html.push_str(&payload);
    html.push_str("</script>\n<script>");
    html.push_str(include_str!("viewer/dagre.min.js"));
    html.push_str("</script>\n<script>");
    html.push_str(include_str!("viewer/layout.js"));
    html.push_str("</script>\n<script>");
    html.push_str(TRACE_JS);
    html.push_str("</script>\n</body>\n</html>\n");
    html
}

/// Serialize JSON and escape so embedded payload cannot close a surrounding `<script>` tag.
pub fn embed_json(value: &Value) -> String {
    escape_for_script(&value.to_string())
}

/// Escape text for safe inclusion inside HTML `<script>` content.
pub fn escape_for_script(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            _ => out.push(ch),
        }
    }
    out
}
