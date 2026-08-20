//! Dependency-free interactive HTML visualizer for `candle-graph/viewer/5` evidence payloads.
//!
//! Embeds escaped JSON, CSS, and JS into one document. No CDN or network fetches.

use serde_json::Value;

use crate::evidence::EvidencePacket;

pub mod trace_view;

const CSS: &str = include_str!("viewer/style.css");
const TRACE_JS: &str = include_str!("viewer/app_trace.js");

/// Render application and GPU evidence in one standalone document.
pub fn render_evidence_html(evidence: &EvidencePacket) -> String {
    let projection = trace_view::project(evidence);
    render_trace_document(&projection)
}

fn render_trace_document(projection: &Value) -> String {
    let payload = embed_json(projection);
    let mut html = String::with_capacity(8192 + CSS.len() + TRACE_JS.len() + payload.len());
    html.push_str(
        "<!DOCTYPE html>\n<html lang=\"en\" data-viewer=\"candle-graph-evidence\">\n<head>\n",
    );
    html.push_str("<meta charset=\"utf-8\"/>\n");
    html.push_str("<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>\n");
    html.push_str("<title>candle-graph evidence</title>\n<style>");
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
.span-cost-table tbody tr{cursor:pointer}\n\
.span-cost-table tbody tr:hover,.span-cost-table tbody tr.sel{background:var(--accent-soft)}\n\
.span-tree .span-row:focus-visible{outline:2px solid var(--focus);outline-offset:-2px}\n\
.node.function .node-card,.node.root .node-card,.node.module .node-card{stroke-width:2.5}\n",
    );
    html.push_str("</style>\n</head>\n<body>\n");
    html.push_str("<a class=\"skip\" href=\"#profile-viewer\">Skip to profile evidence</a>\n");

    html.push_str("<header class=\"top\" role=\"banner\">\n");
    html.push_str("  <div class=\"brand\"><span class=\"brand-mark\" aria-hidden=\"true\"></span>candle-graph evidence</div>\n");
    html.push_str(
        "  <div class=\"cov\" data-coverage role=\"status\" aria-live=\"polite\"></div>\n",
    );
    html.push_str("  <div class=\"top-actions\">\n");
    html.push_str("    <button type=\"button\" class=\"btn primary\" id=\"export-btn\" data-trace-only aria-label=\"Export trace graph as SVG\" hidden>Export SVG</button>\n");
    html.push_str("    <button type=\"button\" class=\"btn\" id=\"theme-btn\" data-theme-toggle aria-label=\"Toggle color theme\">Theme</button>\n");
    html.push_str("  </div>\n");
    html.push_str("</header>\n");

    html.push_str("<div class=\"layout\" id=\"app\">\n");
    html.push_str(
        "  <nav class=\"pane pane-sidebar\" data-pane=\"sidebar\" aria-label=\"Navigation\">\n",
    );
    html.push_str("    <div class=\"pane-h\">Views</div>\n");
    html.push_str("    <div class=\"tabs\" data-view-tabs role=\"tablist\" aria-label=\"Profile evidence views\"></div>\n");
    html.push_str("    <div id=\"trace-navigation\" class=\"trace-navigation\" hidden>\n");
    html.push_str("      <div class=\"pane-h sub\">Span hierarchy</div>\n");
    html.push_str("      <div class=\"search-box\">\n");
    html.push_str("        <label class=\"sr\" for=\"span-search\">Search spans</label>\n");
    html.push_str("        <input id=\"span-search\" data-span-search type=\"search\" placeholder=\"Filter spans…\" autocomplete=\"off\"/>\n");
    html.push_str("      </div>\n");
    html.push_str("      <div id=\"span-tree\" class=\"scroll-area span-tree\" data-span-tree role=\"tree\" aria-label=\"Span hierarchy\"></div>\n");
    html.push_str("    </div>\n");
    html.push_str("  </nav>\n");
    html.push_str(
        "  <div class=\"resize-handle\" data-side=\"left\" aria-hidden=\"true\"></div>\n",
    );

    html.push_str("  <main class=\"pane pane-canvas\" data-pane=\"canvas\" id=\"profile-viewer\" aria-label=\"Profile evidence\">\n");
    html.push_str("    <div class=\"pane-h row\"><span data-canvas-title>Evidence</span><span class=\"graph-stats\" data-graph-stats></span></div>\n");
    html.push_str(
        "    <div class=\"graph-toolbar\" id=\"graph-toolbar\" data-trace-only hidden>\n",
    );
    html.push_str("      <button type=\"button\" class=\"btn\" id=\"fit-btn\" aria-label=\"Fit graph to view\">Fit</button>\n");
    html.push_str("      <button type=\"button\" class=\"btn\" id=\"reset-btn\" aria-label=\"Reset zoom and pan\">Reset</button>\n");
    html.push_str("    </div>\n");
    html.push_str("    <section id=\"view-panel-evidence\" class=\"view-panel scroll-area evidence-view\" data-view-panel=\"evidence\" role=\"tabpanel\" aria-labelledby=\"view-tab-evidence\"></section>\n");
    html.push_str("    <div class=\"canvas-wrap\" id=\"view-panel-trace\" data-view-panel=\"trace\" role=\"tabpanel\" aria-labelledby=\"view-tab-trace\" hidden>\n");
    html.push_str("      <div id=\"empty-graph\" class=\"empty\" data-empty-state hidden>\n");
    html.push_str("        <p><strong>No trace data</strong></p>\n");
    html.push_str("        <p>This trace has no spans to display.</p>\n");
    html.push_str("      </div>\n");
    html.push_str("      <svg id=\"graph-canvas\" data-canvas role=\"img\" aria-label=\"Trace graph\" tabindex=\"0\"></svg>\n");
    html.push_str("      <div id=\"graph-tooltip\" class=\"graph-tooltip\" role=\"tooltip\" aria-hidden=\"true\"></div>\n");
    html.push_str(
        "      <div class=\"legend-float\" data-legend role=\"list\" aria-label=\"Heat legend\">\n",
    );
    html.push_str("        <button type=\"button\" class=\"legend-toggle\" id=\"legend-toggle\" aria-expanded=\"true\">Legend</button>\n");
    html.push_str("        <div class=\"legend-body\">\n");
    html.push_str("          <span role=\"listitem\"><label><input type=\"radio\" name=\"heat-mode\" value=\"time\" checked> Self time heat</label></span>\n");
    html.push_str("          <span role=\"listitem\"><label><input type=\"radio\" name=\"heat-mode\" value=\"memory\"> Memory heat</label></span>\n");
    html.push_str(
        "          <span role=\"listitem\" class=\"lg kind-op\">Call edge (host duration)</span>\n",
    );
    html.push_str(
        "          <span role=\"listitem\" class=\"lg kind-module\">Tensor data edge</span>\n",
    );
    html.push_str("        </div>\n");
    html.push_str("      </div>\n");
    html.push_str("      <div class=\"canvas-controls\" aria-label=\"Zoom controls\">\n");
    html.push_str("        <button type=\"button\" class=\"btn icon\" id=\"zoom-in\" aria-label=\"Zoom in\">+</button>\n");
    html.push_str("        <div class=\"zoom-label\" id=\"zoom-label\">100%</div>\n");
    html.push_str("        <button type=\"button\" class=\"btn icon\" id=\"zoom-out\" aria-label=\"Zoom out\">−</button>\n");
    html.push_str("        <button type=\"button\" class=\"btn icon\" id=\"zoom-fit\" aria-label=\"Fit to view\">⤢</button>\n");
    html.push_str("      </div>\n");
    html.push_str("    </div>\n");
    html.push_str("    <section id=\"view-panel-span_costs\" class=\"view-panel scroll-area\" data-view-panel=\"span_costs\" role=\"tabpanel\" aria-labelledby=\"view-tab-span_costs\" hidden></section>\n");
    html.push_str("    <section id=\"view-panel-memory\" class=\"view-panel scroll-area\" data-view-panel=\"memory\" role=\"tabpanel\" aria-labelledby=\"view-tab-memory\" hidden></section>\n");
    html.push_str("    <section id=\"view-panel-gpu\" class=\"view-panel scroll-area gpu-view\" data-view-panel=\"gpu\" role=\"tabpanel\" aria-labelledby=\"view-tab-gpu\" hidden></section>\n");
    html.push_str("  </main>\n");

    html.push_str(
        "  <div class=\"resize-handle\" data-side=\"right\" aria-hidden=\"true\"></div>\n",
    );
    html.push_str("  <aside class=\"pane pane-inspector\" data-pane=\"inspector\" aria-label=\"Inspector\" hidden>\n");
    html.push_str("    <div class=\"pane-h\">Inspector</div>\n");
    html.push_str("    <dl id=\"inspector\" data-inspector>\n");
    html.push_str("      <div><dt>Label</dt><dd data-field=\"label\">—</dd></div>\n");
    html.push_str("      <div><dt>Kind</dt><dd data-field=\"kind\">—</dd></div>\n");
    html.push_str("      <div><dt>Host self</dt><dd data-field=\"self_time\">—</dd></div>\n");
    html.push_str("      <div><dt>Host total</dt><dd data-field=\"total_time\">—</dd></div>\n");
    html.push_str("      <div><dt>Shape</dt><dd data-field=\"shape\">—</dd></div>\n");
    html.push_str("      <div><dt>Dtype</dt><dd data-field=\"dtype\">—</dd></div>\n");
    html.push_str("      <div><dt>Dense footprint</dt><dd data-field=\"dense\">—</dd></div>\n");
    html.push_str("      <div><dt>Peak live</dt><dd data-field=\"peak_bytes\">—</dd></div>\n");
    html.push_str("      <div><dt>Allocated</dt><dd data-field=\"bytes\">—</dd></div>\n");
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
