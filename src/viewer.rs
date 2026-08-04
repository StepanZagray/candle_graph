//! Dependency-free interactive HTML visualizer for `candle-graph/viewer/1` payloads.
//!
//! Embeds escaped JSON, CSS, and JS into one document. No CDN or network fetches.

use serde_json::Value;

const CSS: &str = include_str!("viewer/style.css");
const JS: &str = include_str!("viewer/app.js");

/// Render a complete standalone HTML document from a viewer projection.
pub fn render_html(projection: &Value) -> String {
    let payload = embed_json(projection);
    let mut html = String::with_capacity(8192 + CSS.len() + JS.len() + payload.len());
    html.push_str("<!DOCTYPE html>\n<html lang=\"en\" data-viewer=\"candle-graph\">\n<head>\n");
    html.push_str("<meta charset=\"utf-8\"/>\n");
    html.push_str("<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>\n");
    html.push_str("<title>candle-graph visualizer</title>\n<style>");
    html.push_str(CSS);
    html.push_str("</style>\n</head>\n<body>\n");
    html.push_str("<a class=\"skip\" href=\"#canvas-pane\">Skip to canvas</a>\n");

    // Header
    html.push_str("<header class=\"top\" role=\"banner\">\n");
    html.push_str("  <div class=\"brand\"><span class=\"brand-mark\" aria-hidden=\"true\"></span>candle-graph</div>\n");
    html.push_str("  <div class=\"cov\" data-coverage role=\"status\" aria-live=\"polite\"></div>\n");
    html.push_str("  <div class=\"top-actions\">\n");
    html.push_str("    <button type=\"button\" class=\"btn\" id=\"print-btn\" aria-label=\"Toggle publication preview\">Preview</button>\n");
    html.push_str("    <button type=\"button\" class=\"btn primary\" id=\"export-btn\" aria-label=\"Export graph as SVG\">Export SVG</button>\n");
    html.push_str("    <button type=\"button\" class=\"btn\" id=\"theme-btn\" data-theme-toggle aria-label=\"Toggle color theme\">Theme</button>\n");
    html.push_str("  </div>\n");
    html.push_str("</header>\n");

    // Layout with resizable panes
    html.push_str("<div class=\"layout\" id=\"app\">\n");

    // Sidebar
    html.push_str("  <nav class=\"pane pane-sidebar\" data-pane=\"sidebar\" aria-label=\"Navigation\">\n");
    html.push_str("    <div class=\"pane-h\">Views</div>\n");
    html.push_str("    <div class=\"tabs\" data-view-tabs role=\"tablist\" aria-label=\"Graph views\"></div>\n");
    html.push_str("    <div class=\"search-box\" id=\"search-section\">\n");
    html.push_str("      <label class=\"sr\" for=\"node-search\">Search graph nodes</label>\n");
    html.push_str("      <input id=\"node-search\" type=\"search\" placeholder=\"Search nodes…\" autocomplete=\"off\"/>\n");
    html.push_str("    </div>\n");
    html.push_str("    <div id=\"tree-section\">\n");
    html.push_str("      <div class=\"pane-h sub\">Module tree</div>\n");
    html.push_str("      <div class=\"search-box\">\n");
    html.push_str("        <label class=\"sr\" for=\"mod-search\">Search modules</label>\n");
    html.push_str("        <input id=\"mod-search\" data-module-search type=\"search\" placeholder=\"Filter modules…\" autocomplete=\"off\"/>\n");
    html.push_str("      </div>\n");
    html.push_str("      <div id=\"module-tree\" class=\"scroll-area\" data-module-tree tabindex=\"0\" role=\"tree\" aria-label=\"Modules\"></div>\n");
    html.push_str("    </div>\n");
    html.push_str("    <div class=\"pane-h sub\">Findings</div>\n");
    html.push_str("    <div id=\"findings-list\" class=\"scroll-area\" data-findings-list role=\"list\" aria-label=\"Findings\"></div>\n");
    html.push_str("    <div class=\"diag\" data-diagnostics aria-label=\"Diagnostics\"></div>\n");
    html.push_str("  </nav>\n");

    html.push_str("  <div class=\"resize-handle\" data-side=\"left\" aria-hidden=\"true\"></div>\n");

    // Canvas
    html.push_str("  <main class=\"pane pane-canvas\" data-pane=\"canvas\" id=\"canvas-pane\" aria-label=\"Graph canvas\">\n");
    html.push_str("    <div class=\"pane-h row\"><span data-canvas-title>Graph</span><span class=\"graph-stats\" data-graph-stats></span></div>\n");
    html.push_str("    <div class=\"graph-toolbar\" id=\"graph-toolbar\" hidden>\n");
    html.push_str("      <label class=\"sr\" for=\"fn-filter\">Function</label>\n");
    html.push_str("      <select id=\"fn-filter\" data-function-filter aria-label=\"Filter by function\"></select>\n");
    html.push_str("      <button type=\"button\" class=\"btn\" id=\"fit-btn\" aria-label=\"Fit graph to view\">Fit</button>\n");
    html.push_str("      <button type=\"button\" class=\"btn\" id=\"reset-btn\" aria-label=\"Reset zoom and pan\">Reset</button>\n");
    html.push_str("    </div>\n");
    html.push_str("    <div class=\"graph-banner\" id=\"graph-banner\" hidden role=\"status\"></div>\n");
    html.push_str("    <div class=\"canvas-wrap\" id=\"canvas-wrap\">\n");
    html.push_str("      <div id=\"empty-graph\" class=\"empty\" data-empty-state hidden>\n");
    html.push_str("        <p><strong>No graph data</strong></p>\n");
    html.push_str("        <p>This view has no nodes to display for the current analysis.</p>\n");
    html.push_str("      </div>\n");
    html.push_str("      <svg id=\"graph-canvas\" data-canvas role=\"img\" aria-label=\"Graph\" tabindex=\"0\"></svg>\n");
    html.push_str("      <div id=\"graph-tooltip\" class=\"graph-tooltip\" role=\"tooltip\" aria-hidden=\"true\"></div>\n");
    html.push_str("      <div class=\"legend-float\" data-legend role=\"list\" aria-label=\"Graph legend\">\n");
    html.push_str("        <button type=\"button\" class=\"legend-toggle\" id=\"legend-toggle\" aria-expanded=\"true\">Legend</button>\n");
    html.push_str("        <div class=\"legend-body\">\n");
    html.push_str("          <span role=\"listitem\" class=\"lg kind-op\">Operation</span>\n");
    html.push_str("          <span role=\"listitem\" class=\"lg kind-param\">Parameter</span>\n");
    html.push_str("          <span role=\"listitem\" class=\"lg kind-tensor\">Tensor</span>\n");
    html.push_str("          <span role=\"listitem\" class=\"lg kind-module\">Module</span>\n");
    for state in [
        "Trainable",
        "Frozen",
        "Differentiable",
        "Severed",
        "LayoutDependent",
        "Unknown",
    ] {
        html.push_str(&format!(
            "          <span role=\"listitem\" data-legend-item=\"{state}\" class=\"lg {state}\">{state}</span>\n"
        ));
    }
    html.push_str("        </div>\n");
    html.push_str("      </div>\n");
    html.push_str("      <div class=\"canvas-controls\" aria-label=\"Zoom controls\">\n");
    html.push_str("        <button type=\"button\" class=\"btn icon\" id=\"zoom-in\" aria-label=\"Zoom in\" title=\"Zoom in (+)\">+</button>\n");
    html.push_str("        <div class=\"zoom-label\" id=\"zoom-label\">100%</div>\n");
    html.push_str("        <button type=\"button\" class=\"btn icon\" id=\"zoom-out\" aria-label=\"Zoom out\" title=\"Zoom out (−)\">−</button>\n");
    html.push_str("        <button type=\"button\" class=\"btn icon\" id=\"zoom-fit\" aria-label=\"Fit to view\" title=\"Fit (F)\">⤢</button>\n");
    html.push_str("      </div>\n");
    html.push_str("      <p class=\"canvas-hint\" id=\"canvas-hint\"><kbd>U</kbd> clear · click background · <kbd>F</kbd> fit · <kbd>+</kbd><kbd>−</kbd> zoom</p>\n");
    html.push_str("    </div>\n");
    html.push_str("  </main>\n");

    html.push_str("  <div class=\"resize-handle\" data-side=\"right\" aria-hidden=\"true\"></div>\n");

    // Inspector
    html.push_str("  <aside class=\"pane pane-inspector\" data-pane=\"inspector\" aria-label=\"Inspector\">\n");
    html.push_str("    <div class=\"pane-h\">Inspector</div>\n");
    html.push_str("    <dl id=\"inspector\" data-inspector>\n");
    html.push_str("      <div class=\"section-title\">Identity</div>\n");
    for (field, title) in [
        ("label", "Label"),
        ("kind", "Kind"),
        ("qualified", "Qualified name"),
        ("source", "Source"),
    ] {
        html.push_str(&format!(
            "      <div><dt>{title}</dt><dd data-field=\"{field}\">—</dd></div>\n"
        ));
    }
    html.push_str("      <div class=\"section-title\">Tensor</div>\n");
    for (field, title) in [
        ("shape", "Shape"),
        ("dtype", "Dtype"),
        ("grad", "Grad state"),
        ("role", "Role"),
    ] {
        html.push_str(&format!(
            "      <div><dt>{title}</dt><dd data-field=\"{field}\">—</dd></div>\n"
        ));
    }
    html.push_str("      <div class=\"section-title\">Analysis</div>\n");
    for (field, title) in [
        ("root", "Builder root"),
        ("confidence", "Confidence"),
        ("timing", "Timing (avg · min–max)"),
        ("function", "Function"),
        ("severity", "Severity"),
        ("rule", "Rule"),
    ] {
        html.push_str(&format!(
            "      <div><dt>{title}</dt><dd data-field=\"{field}\">—</dd></div>\n"
        ));
    }
    html.push_str("    </dl>\n");
    html.push_str("  </aside>\n");
    html.push_str("</div>\n");

    html.push_str("<script id=\"cg-payload\" type=\"application/json\">");
    html.push_str(&payload);
    html.push_str("</script>\n<script>");
    html.push_str(include_str!("viewer/dagre.min.js"));
    html.push_str("</script>\n<script>");
    html.push_str(include_str!("viewer/layout.js"));
    html.push_str("</script>\n<script>");
    html.push_str(JS);
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
