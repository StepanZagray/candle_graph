//! Dependency-free interactive HTML viewer for structure (+ optional dataflow) JSON.
//!
//! Embeds escaped JSON, CSS, and JS into one document. No CDN or network fetches.

use serde_json::Value;

/// Render a complete standalone HTML document from a structure report and optional dataflow JSON.
pub fn render_html(structure: &Value, dataflow: Option<&Value>) -> String {
    let structure_json = embed_json(structure);
    let dataflow_json = match dataflow {
        Some(v) => embed_json(v),
        None => "null".to_string(),
    };
    let empty_attr = if dataflow.is_some() {
        ""
    } else {
        r#" data-empty-dataflow="true""#
    };

    // Avoid `format!` — CSS/JS are brace-heavy. Concatenate around escaped payloads.
    let mut html = String::with_capacity(
        4096 + CSS.len() + JS.len() + structure_json.len() + dataflow_json.len(),
    );
    html.push_str("<!DOCTYPE html>\n<html lang=\"en\" data-viewer=\"candle-graph\">\n<head>\n");
    html.push_str("<meta charset=\"utf-8\"/>\n");
    html.push_str("<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>\n");
    html.push_str("<title>candle-graph viewer</title>\n<style>");
    html.push_str(CSS);
    html.push_str("</style>\n</head>\n<body>\n");
    html.push_str("<a class=\"skip\" href=\"#canvas-pane\">Skip to canvas</a>\n");
    html.push_str("<header class=\"top\" role=\"banner\">\n");
    html.push_str("  <div class=\"brand\">candle-graph</div>\n");
    html.push_str(
        "  <div class=\"cov\" data-coverage role=\"status\" aria-live=\"polite\"></div>\n",
    );
    html.push_str("  <button type=\"button\" id=\"theme-btn\" data-theme-toggle aria-label=\"Toggle color theme\">Theme</button>\n");
    html.push_str("</header>\n<div class=\"layout\" id=\"app\"");
    html.push_str(empty_attr);
    html.push_str(">\n");
    html.push_str(
        "  <nav class=\"pane\" data-pane=\"hierarchy\" aria-label=\"Module hierarchy\">\n",
    );
    html.push_str("    <div class=\"pane-h\">Modules</div>\n");
    html.push_str("    <label class=\"sr\" for=\"mod-search\">Search modules</label>\n");
    html.push_str("    <input id=\"mod-search\" data-module-search type=\"search\" placeholder=\"Search…\" autocomplete=\"off\"/>\n");
    html.push_str("    <div id=\"module-tree\" data-module-tree tabindex=\"0\" role=\"tree\" aria-label=\"Modules\"></div>\n");
    html.push_str("    <div class=\"diag\" data-diagnostics aria-label=\"Diagnostics\"></div>\n");
    html.push_str("  </nav>\n");
    html.push_str("  <main class=\"pane\" data-pane=\"canvas\" id=\"canvas-pane\" aria-label=\"Dataflow canvas\">\n");
    html.push_str("    <div class=\"pane-h\">Dataflow</div>\n");
    html.push_str("    <div class=\"canvas-wrap\" id=\"canvas-wrap\">\n");
    html.push_str("      <div id=\"empty-dataflow\" class=\"empty\" data-empty-state hidden>\n");
    html.push_str("        <p><strong>No dataflow graph</strong></p>\n");
    html.push_str("        <p>Structure is shown in the module tree. Pass dataflow JSON to visualize tensors, ops, and gradient state.</p>\n");
    html.push_str("      </div>\n");
    html.push_str("      <svg id=\"dataflow-canvas\" data-canvas role=\"img\" aria-label=\"Dataflow graph\" tabindex=\"0\"></svg>\n");
    html.push_str("    </div>\n");
    html.push_str("    <div class=\"legend\" data-legend role=\"list\" aria-label=\"Gradient state legend\">\n");
    html.push_str("      <span role=\"listitem\" data-legend-item=\"Trainable\" class=\"lg Trainable\">Trainable</span>\n");
    html.push_str("      <span role=\"listitem\" data-legend-item=\"Frozen\" class=\"lg Frozen\">Frozen</span>\n");
    html.push_str("      <span role=\"listitem\" data-legend-item=\"Differentiable\" class=\"lg Differentiable\">Differentiable</span>\n");
    html.push_str("      <span role=\"listitem\" data-legend-item=\"Severed\" class=\"lg Severed\">Severed</span>\n");
    html.push_str("      <span role=\"listitem\" data-legend-item=\"LayoutDependent\" class=\"lg LayoutDependent\">LayoutDependent</span>\n");
    html.push_str("      <span role=\"listitem\" data-legend-item=\"Unknown\" class=\"lg Unknown\">Unknown</span>\n");
    html.push_str("    </div>\n");
    html.push_str("  </main>\n");
    html.push_str("  <aside class=\"pane\" data-pane=\"inspector\" aria-label=\"Inspector\">\n");
    html.push_str("    <div class=\"pane-h\">Inspector</div>\n");
    html.push_str("    <dl id=\"inspector\" data-inspector>\n");
    html.push_str("      <div><dt>Source</dt><dd data-field=\"source\">—</dd></div>\n");
    html.push_str("      <div><dt>Shape</dt><dd data-field=\"shape\">—</dd></div>\n");
    html.push_str("      <div><dt>Dtype</dt><dd data-field=\"dtype\">—</dd></div>\n");
    html.push_str("      <div><dt>Builder root</dt><dd data-field=\"root\">—</dd></div>\n");
    html.push_str("      <div><dt>Certainty</dt><dd data-field=\"certainty\">—</dd></div>\n");
    html.push_str("      <div><dt>Grad state</dt><dd data-field=\"grad\">—</dd></div>\n");
    html.push_str("      <div><dt>Label</dt><dd data-field=\"label\">—</dd></div>\n");
    html.push_str("      <div><dt>Kind</dt><dd data-field=\"kind\">—</dd></div>\n");
    html.push_str("    </dl>\n");
    html.push_str("  </aside>\n");
    html.push_str("</div>\n");
    html.push_str("<script id=\"cg-structure\" type=\"application/json\">");
    html.push_str(&structure_json);
    html.push_str("</script>\n<script id=\"cg-dataflow\" type=\"application/json\">");
    html.push_str(&dataflow_json);
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
///
/// Replaces `<` with a JSON unicode escape so `</script>` in source strings cannot break out.
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

const CSS: &str = r#"
:root{
  --bg:#f4f6f8;--fg:#1a1f24;--muted:#5a6570;--pane:#fff;--border:#c9d1d8;
  --accent:#0b6e4f;--focus:#0b57d0;--hazard:#a33;--canvas:#eef2f5;
  --Trainable:#0b6e4f;--Frozen:#5a6570;--Differentiable:#0b57d0;
  --Severed:#a33;--LayoutDependent:#9a6700;--Unknown:#6e40c9;
}
[data-theme=dark]{
  --bg:#12161a;--fg:#e7ecf1;--muted:#9aa7b2;--pane:#1b2229;--border:#31404c;
  --accent:#3dd68c;--focus:#8ab4f8;--hazard:#ff8a80;--canvas:#0f1418;
  --Trainable:#3dd68c;--Frozen:#9aa7b2;--Differentiable:#8ab4f8;
  --Severed:#ff8a80;--LayoutDependent:#f0c000;--Unknown:#c4a7ff;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;font:14px/1.4 "IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--fg)}
.skip{position:absolute;left:-999px;top:0;background:var(--pane);padding:.5rem;z-index:9}
.skip:focus{left:0}
.top{display:flex;gap:.75rem;align-items:center;padding:.5rem .75rem;border-bottom:1px solid var(--border);background:var(--pane)}
.brand{font-weight:700;letter-spacing:.02em}
.cov{flex:1;color:var(--muted);font-size:12px}
#theme-btn{border:1px solid var(--border);background:var(--bg);color:var(--fg);padding:.25rem .6rem;cursor:pointer}
.layout{display:grid;grid-template-columns:minmax(200px,1fr) minmax(280px,2.2fr) minmax(200px,1fr);gap:0;height:calc(100% - 42px);min-height:320px}
.pane{display:flex;flex-direction:column;min-width:0;min-height:0;border-right:1px solid var(--border);background:var(--pane)}
.pane:last-child{border-right:0}
.pane-h{font-weight:600;padding:.5rem .75rem;border-bottom:1px solid var(--border)}
#mod-search{margin:.5rem .75rem;padding:.35rem .5rem;border:1px solid var(--border);background:var(--bg);color:var(--fg)}
#module-tree{overflow:auto;flex:1;padding:.25rem .5rem .75rem;outline:none}
#module-tree:focus{box-shadow:inset 0 0 0 2px var(--focus)}
.tree-item{display:flex;align-items:center;gap:.25rem;padding:.15rem .25rem;border-radius:2px;cursor:pointer}
.tree-item:hover,.tree-item[aria-selected=true]{background:color-mix(in srgb,var(--accent) 16%,transparent)}
.tree-item:focus{outline:2px solid var(--focus);outline-offset:-2px}
.tw{border:0;background:transparent;color:var(--muted);width:1.1rem;cursor:pointer}
.tn{color:var(--muted);font-size:12px;margin-left:.25rem}
.diag{max-height:30%;overflow:auto;border-top:1px solid var(--border);padding:.5rem .75rem;font-size:12px;color:var(--hazard)}
.diag:empty{display:none}
.canvas-wrap{position:relative;flex:1;min-height:200px;background:var(--canvas);overflow:hidden}
#dataflow-canvas{width:100%;height:100%;touch-action:none;cursor:grab}
#dataflow-canvas:active{cursor:grabbing}
#dataflow-canvas:focus{outline:2px solid var(--focus);outline-offset:-2px}
.empty{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:1.5rem;text-align:center;color:var(--muted);z-index:1;pointer-events:none}
.empty[hidden]{display:none}
.legend{display:flex;flex-wrap:wrap;gap:.4rem .75rem;padding:.4rem .75rem;border-top:1px solid var(--border);font-size:12px}
.lg::before{content:"";display:inline-block;width:.65rem;height:.65rem;margin-right:.3rem;border-radius:50%;background:var(--c,var(--muted));vertical-align:middle}
.lg.Trainable{--c:var(--Trainable)}.lg.Frozen{--c:var(--Frozen)}.lg.Differentiable{--c:var(--Differentiable)}
.lg.Severed{--c:var(--Severed)}.lg.LayoutDependent{--c:var(--LayoutDependent)}.lg.Unknown{--c:var(--Unknown)}
#inspector{margin:0;padding:.5rem .75rem;overflow:auto;flex:1}
#inspector>div{display:grid;grid-template-columns:7rem 1fr;gap:.25rem .5rem;padding:.3rem 0;border-bottom:1px solid var(--border)}
dt{color:var(--muted);font-weight:500}dd{margin:0;word-break:break-word}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
.node{cursor:pointer}.node.sel rect,.edge.sel{stroke:var(--focus);stroke-width:2.5}
.edge{cursor:pointer;fill:none;stroke-width:1.5}
@media(max-width:900px){
  .layout{grid-template-columns:1fr;grid-template-rows:minmax(160px,auto) minmax(240px,1fr) minmax(160px,auto);height:auto}
  .pane{border-right:0;border-bottom:1px solid var(--border);max-height:50vh}
  .canvas-wrap{min-height:280px}
}
"#;

const JS: &str = r#"
(function(){
const S=JSON.parse(document.getElementById('cg-structure').textContent);
const D=JSON.parse(document.getElementById('cg-dataflow').textContent);
const GRAD=['Trainable','Frozen','Differentiable','Severed','LayoutDependent','Unknown'];
const root=document.documentElement;
const pref=matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';
root.setAttribute('data-theme',localStorage.getItem('cg-theme')||pref);
document.getElementById('theme-btn').onclick=()=>{
  const n=root.getAttribute('data-theme')==='dark'?'light':'dark';
  root.setAttribute('data-theme',n);localStorage.setItem('cg-theme',n);
};
function cert(c){
  if(c==null)return '—';
  if(typeof c==='string')return c;
  if(c.kind)return c.reason?c.kind+': '+c.reason:c.kind;
  return String(c);
}
function cov(){
  const c=S.coverage||{};
  const el=document.querySelector('[data-coverage]');
  el.textContent=[
    (c.instances!=null?c.instances+' instances':null),
    (c.params!=null?c.params+' params':null),
    (c.params_certain!=null?c.params_certain+' certain':null),
    (c.params_conditional!=null?c.params_conditional+' conditional':null),
    (c.params_unknown!=null?c.params_unknown+' unknown':null),
    (c.diagnostics!=null?c.diagnostics+' diagnostics':null)
  ].filter(Boolean).join(' · ')||'No coverage fields';
  const box=document.querySelector('[data-diagnostics]');
  const diags=(S.diagnostics||[]).concat((D&&D.diagnostics)||[]);
  box.innerHTML=diags.length?diags.map(d=>{
    const m=esc(d.message||d||'');
    const at=d.at?esc(String(d.at))+' — ':'';
    return '<div>'+at+m+'</div>';
  }).join(''):'';
}
function esc(s){return String(s).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));}
function setInspector(o){
  o=o||{};
  const map={
    source:o.at||o.source||o.span||'—',
    shape:o.shape!=null?(Array.isArray(o.shape)?o.shape.join('×'):String(o.shape)):'—',
    dtype:o.dtype||'—',
    root:o.root||o.builder_root||'—',
    certainty:cert(o.certainty),
    grad:o.grad_state||o.grad||'—',
    label:o.label||o.key||o.type||o.name||'—',
    kind:o.kind||o.op||'—'
  };
  Object.keys(map).forEach(k=>{
    const el=document.querySelector('[data-field="'+k+'"]');
    if(el)el.textContent=map[k];
  });
}
function buildTree(){
  const mods=S.modules||[];
  const params=S.parameters||[];
  const byParent={};
  mods.forEach((m,i)=>{
    const p=m.parent_id==null?'root':String(m.parent_id);
    (byParent[p]=byParent[p]||[]).push({m,i});
  });
  const tree=document.getElementById('module-tree');
  const open=new Set();
  function kids(m){return byParent[String(m.id!=null?m.id:'legacy-'+m.prefix)]||[];}
  function roots(){return byParent.root||mods.map((m,i)=>({m,i})).filter(({m})=>m.parent==null);}
  function render(){
    const q=(document.getElementById('mod-search').value||'').toLowerCase();
    tree.innerHTML='';
    function subtreeHit(m,seen){
      const id=String(m.id!=null?m.id:'legacy-'+m.prefix);
      if(seen.has(id))return false;
      seen.add(id);
      const label=(m.field?m.field+': ':'')+(m.type||m.prefix||'module');
      const selfHit=!q||label.toLowerCase().includes(q)||String(m.prefix||'').toLowerCase().includes(q)||String(m.root||'').toLowerCase().includes(q);
      return selfHit||kids(m).some(({m:child})=>subtreeHit(child,new Set(seen)));
    }
    function add(items,depth,seen){
      items.forEach(({m,i})=>{
        const id=String(m.id!=null?m.id:'legacy-'+m.prefix);
        if(seen.has(id))return;
        const branchSeen=new Set(seen);branchSeen.add(id);
        const label=(m.field?m.field+': ':'')+(m.type||m.prefix||'module');
        const path=String(m.prefix||'');
        const hit=!q||label.toLowerCase().includes(q)||path.toLowerCase().includes(q)||String(m.root||'').toLowerCase().includes(q);
        const ch=kids(m);
        const has=ch.length>0;
        if(!hit&&!subtreeHit(m,new Set()))return;
        const row=document.createElement('div');
        row.className='tree-item';
        row.setAttribute('role','treeitem');
        row.setAttribute('aria-level',String(depth+1));
        row.tabIndex=-1;
        row.dataset.idx=String(i);row.dataset.moduleId=id;
        row.style.paddingLeft=(depth*12+4)+'px';
        const expanded=open.has(id)||!!q;
        if(has){
          row.setAttribute('aria-expanded',expanded?'true':'false');
          const b=document.createElement('button');
          b.type='button';b.className='tw';b.setAttribute('aria-label',expanded?'Collapse':'Expand');
          b.textContent=expanded?'▾':'▸';
          b.onclick=e=>{e.stopPropagation();if(open.has(id))open.delete(id);else open.add(id);render();};
          row.appendChild(b);
        }else{
          const sp=document.createElement('span');sp.className='tw';sp.textContent='·';row.appendChild(sp);
        }
        const t=document.createElement('span');t.textContent=label;row.appendChild(t);
        const meta=document.createElement('span');meta.className='tn';meta.textContent=(m.root||'')+(m.repeat?' ×':'');row.appendChild(meta);
        row.onclick=()=>{
          [...tree.querySelectorAll('[aria-selected=true]')].forEach(x=>x.setAttribute('aria-selected','false'));
          row.setAttribute('aria-selected','true');
          const owned=params.filter(p=>p.module_prefix===path||p.module===m.type);
          setInspector(Object.assign({},m,{label:label,kind:'module',shape:owned.length?owned.length+' params':undefined}));
        };
        tree.appendChild(row);
        if(has&&expanded)add(ch,depth+1,branchSeen);
      });
    }
    add(roots(),0,new Set());
  }
  document.getElementById('mod-search').oninput=render;
  tree.addEventListener('keydown',e=>{
    const items=[...tree.querySelectorAll('.tree-item')];
    const cur=document.activeElement;
    let i=items.indexOf(cur);
    if(e.key==='ArrowDown'){e.preventDefault();(items[i+1]||items[0]||tree).focus();}
    else if(e.key==='ArrowUp'){e.preventDefault();(items[i-1]||items[items.length-1]||tree).focus();}
    else if(e.key==='Enter'&&cur&&cur.classList.contains('tree-item'))cur.click();
    else if(e.key==='Home'){e.preventDefault();(items[0]||tree).focus();}
    else if(e.key==='End'){e.preventDefault();(items[items.length-1]||tree).focus();}
  });
  render();
}
function canvas(){
  const svg=document.getElementById('dataflow-canvas');
  const empty=document.getElementById('empty-dataflow');
  if(!D||!(D.nodes||D.tensors||[]).length){
    empty.hidden=false;svg.setAttribute('aria-hidden','true');return;
  }
  empty.hidden=true;
  const nodes=(D.nodes||D.tensors||[]).map((n,i)=>Object.assign({},n,{_id:String(n.id!=null?n.id:i)}));
  const edges=(D.edges||D.ops||[]).map((e,i)=>Object.assign({},e,{
    _id:String(e.id!=null?e.id:'e'+i),
    _from:String(e.from!=null?e.from:e.src!=null?e.src:e.source),
    _to:String(e.to!=null?e.to:e.dst!=null?e.dst:e.target)
  }));
  // Collapse repeated module groups by default (shared repeat/group/template key).
  const groups={};
  nodes.forEach(n=>{
    const g=n.repeat_group||n.group||(n.repeat&&(n.module||n.label))||(n.template)||null;
    if(g){(groups[g]=groups[g]||[]).push(n);}
  });
  const collapsed=new Set(Object.keys(groups));
  let view={x:0,y:0,k:1};
  function visibleNodes(){
    const hide=new Set();
    const reps=[];
    collapsed.forEach(g=>{
      const arr=groups[g];if(!arr)return;
      arr.slice(1).forEach(n=>hide.add(n._id));
      const r=Object.assign({},arr[0],{
        label:(arr[0].label||arr[0].name||g)+' ×'+arr.length,
        _group:g,_collapsed:true
      });
      reps.push(r);hide.add(arr[0]._id);
    });
    return nodes.filter(n=>!hide.has(n._id)).concat(reps);
  }
  function layout(vis){
    const cols=Math.max(1,Math.ceil(Math.sqrt(vis.length)));
    vis.forEach((n,i)=>{
      n._x=(i%cols)*180+40;n._y=Math.floor(i/cols)*90+40;
      n._w=150;n._h=48;
    });
  }
  function gradOf(n){
    const g=n.grad_state||n.grad||'Unknown';
    return GRAD.indexOf(g)>=0?g:'Unknown';
  }
  function draw(){
    const vis=visibleNodes();layout(vis);
    const byId={};vis.forEach(n=>byId[n._id]=n);
    // Map hidden members to representative.
    const remap={};
    collapsed.forEach(g=>{
      const arr=groups[g];if(!arr)return;
      arr.forEach(n=>{remap[n._id]=arr[0]._id;});
    });
    while(svg.firstChild)svg.removeChild(svg.firstChild);
    const g=document.createElementNS(svg.namespaceURI,'g');
    g.setAttribute('transform','translate('+view.x+','+view.y+') scale('+view.k+')');
    svg.appendChild(g);
    edges.forEach(e=>{
      const a=byId[remap[e._from]||e._from],b=byId[remap[e._to]||e._to];
      if(!a||!b||a===b)return;
      const p=document.createElementNS(svg.namespaceURI,'path');
      const x1=a._x+a._w,y1=a._y+a._h/2,x2=b._x,y2=b._y+b._h/2;
      p.setAttribute('d','M'+x1+' '+y1+' C'+(x1+40)+' '+y1+','+(x2-40)+' '+y2+','+x2+' '+y2);
      p.setAttribute('class','edge');
      p.dataset.edgeId=e._id;
      const sev=(e.grad_state||e.kind||'').toString().toLowerCase().includes('sever');
      p.setAttribute('stroke',sev?'var(--Severed)':'var(--muted)');
      p.setAttribute('marker-end','');
      p.onclick=ev=>{ev.stopPropagation();selectEdge(e,p);};
      g.appendChild(p);
    });
    vis.forEach(n=>{
      const gg=document.createElementNS(svg.namespaceURI,'g');
      gg.setAttribute('class','node');gg.dataset.nodeId=n._id;
      gg.setAttribute('tabindex','0');gg.setAttribute('role','button');
      gg.setAttribute('aria-label',n.label||n._id);
      const r=document.createElementNS(svg.namespaceURI,'rect');
      r.setAttribute('x',n._x);r.setAttribute('y',n._y);r.setAttribute('width',n._w);r.setAttribute('height',n._h);
      r.setAttribute('rx',6);r.setAttribute('fill','var(--pane)');r.setAttribute('stroke','var(--'+gradOf(n)+')');
      const t=document.createElementNS(svg.namespaceURI,'text');
      t.setAttribute('x',n._x+8);t.setAttribute('y',n._y+20);
      t.setAttribute('fill','var(--fg)');t.setAttribute('font-size','12');
      t.textContent=String(n.label||n.key||n.op||n._id).slice(0,22);
      const t2=document.createElementNS(svg.namespaceURI,'text');
      t2.setAttribute('x',n._x+8);t2.setAttribute('y',n._y+38);
      t2.setAttribute('fill','var(--muted)');t2.setAttribute('font-size','11');
      t2.textContent=gradOf(n)+(n.dtype?' · '+n.dtype:'');
      gg.appendChild(r);gg.appendChild(t);gg.appendChild(t2);
      gg.onclick=ev=>{
        ev.stopPropagation();
        if(n._collapsed&&n._group){collapsed.delete(n._group);draw();return;}
        selectNode(n,gg);
      };
      gg.onkeydown=ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();gg.onclick(ev);}};
      g.appendChild(gg);
    });
  }
  function clearSel(){[...svg.querySelectorAll('.sel')].forEach(x=>x.classList.remove('sel'));}
  function selectNode(n,el){clearSel();el.classList.add('sel');setInspector(n);}
  function selectEdge(e,el){clearSel();el.classList.add('sel');setInspector(Object.assign({kind:'edge',label:e._id},e));}
  let drag=null;
  svg.addEventListener('pointerdown',e=>{
    if(e.target.closest('.node,.edge'))return;
    drag={x:e.clientX-view.x,y:e.clientY-view.y};svg.setPointerCapture(e.pointerId);
  });
  svg.addEventListener('pointermove',e=>{if(!drag)return;view.x=e.clientX-drag.x;view.y=e.clientY-drag.y;draw();});
  svg.addEventListener('pointerup',()=>{drag=null;});
  svg.addEventListener('wheel',e=>{
    e.preventDefault();
    const f=e.deltaY<0?1.1:0.9;view.k=Math.min(4,Math.max(0.25,view.k*f));draw();
  },{passive:false});
  svg.addEventListener('keydown',e=>{
    if(e.key==='+'||e.key==='='){view.k=Math.min(4,view.k*1.1);draw();}
    if(e.key==='-'){view.k=Math.max(0.25,view.k/1.1);draw();}
    if(e.key==='0'){view={x:0,y:0,k:1};draw();}
    if(e.key==='Escape'){clearSel();setInspector(null);}
  });
  draw();
}
cov();buildTree();canvas();
})();
"#;
