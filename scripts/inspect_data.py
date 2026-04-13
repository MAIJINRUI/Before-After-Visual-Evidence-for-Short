#!/usr/bin/env python3
"""
inspect_data.py — Human Review Tool for DASE3156 Dataset
=========================================================
Usage:
    python scripts/inspect_data.py [--port 8888]
Then open http://localhost:8888
"""

import json, sys, argparse, mimetypes, urllib.parse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IMGS = DATA / "images"

# ── Load dataset ────────────────────────────────────────
def load():
    # Primary data source: image_manifest.json (it's a LIST)
    manifest_raw = json.loads((DATA / "image_manifest.json").read_text())

    rows = []
    for item in manifest_raw:
        sid = item["sample_id"]
        trial = item.get("trial", "")

        # Build image paths: trial/filename
        # Image files: 0000_before.jpg, 0000_after.jpg (strip 'S' prefix from sample_id)
        num = sid.lstrip("S")  # "S0000" → "0000"
        before_img = f"{num}_before.jpg"
        after_img  = f"{num}_after.jpg"

        row = {
            "sample_id":        sid,
            "split":            item.get("split", "?"),
            "subtype":          item.get("subtype", ""),
            "trial":            trial,
            "before_image_name":item.get("before_image_name", ""),
            "after_image_name": item.get("after_image_name", ""),
            "step_idx":         item.get("step_idx", ""),
            "image_step_idx":   item.get("image_step_idx", ""),
            "same_frame":       item.get("same_frame", ""),
            "quality_score":    item.get("quality_score", ""),
            "_img_before":      before_img,
            "_img_after":       after_img,
        }

        # Try to merge extra fields from samples.jsonl if it exists
        rows.append(row)

    # Optionally merge samples.jsonl
    sjl = DATA / "samples.jsonl"
    if sjl.is_file():
        extra = {}
        with open(sjl) as f:
            for ln in f:
                if ln.strip():
                    obj = json.loads(ln)
                    extra[obj.get("sample_id", "")] = obj
        for r in rows:
            if r["sample_id"] in extra:
                for k, v in extra[r["sample_id"]].items():
                    if k not in r:
                        r[k] = v

    # Optionally merge split.json
    spf = DATA / "split.json"
    if spf.is_file():
        try:
            splits = json.loads(spf.read_text())
            sm = {}
            if isinstance(splits, dict):
                for k, ids in splits.items():
                    if isinstance(ids, list):
                        for i in ids:
                            sm[i] = k
            elif isinstance(splits, list):
                for item in splits:
                    sm[item.get("sample_id", "")] = item.get("split", "?")
            for r in rows:
                if r["sample_id"] in sm:
                    r["split"] = sm[r["sample_id"]]
        except Exception:
            pass

    return rows

print("📦 Loading dataset …")
SAMPLES = load()

# Collect all unique subtypes for filter buttons
ALL_SUBTYPES = sorted(set(r.get("subtype", "") for r in SAMPLES if r.get("subtype")))
print(f"   Subtypes found: {ALL_SUBTYPES}")

JS_DATA = json.dumps(SAMPLES, ensure_ascii=False).replace("</", "<\\/")
JS_SUBTYPES = json.dumps(ALL_SUBTYPES, ensure_ascii=False)
print(f"   ✅ {len(SAMPLES)} samples ready\n")

# ── HTML Page ───────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dataset Inspector — DASE3156</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  background:#f0f2f5;color:#333;height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* ── Top bar ── */
.top{background:linear-gradient(135deg,#1a1a2e,#16213e);color:#fff;padding:10px 20px;
  display:flex;align-items:center;gap:16px;flex-shrink:0;box-shadow:0 2px 8px rgba(0,0,0,.2);
  flex-wrap:wrap}
.top h1{font-size:17px;white-space:nowrap}
.flt{display:flex;gap:5px;flex-wrap:wrap;align-items:center}
.flt button{padding:3px 10px;border:1px solid rgba(255,255,255,.25);border-radius:4px;
  background:transparent;color:#ccc;cursor:pointer;font-size:11px;transition:.15s}
.flt button:hover{border-color:rgba(255,255,255,.5);color:#fff}
.flt button.on{background:#e94560;border-color:#e94560;color:#fff}
.sep{color:rgba(255,255,255,.15);margin:0 4px;user-select:none}
.lbl{font-size:10px;color:rgba(255,255,255,.4);margin-right:2px}
.top input{padding:5px 10px;border-radius:4px;border:none;font-size:12px;width:160px}
.cnt{font-size:12px;opacity:.6;margin-left:auto;white-space:nowrap}

/* ── Layout ── */
.main{display:flex;flex:1;overflow:hidden}

/* ── Sidebar ── */
.side{width:290px;background:#fff;border-right:1px solid #e0e0e0;display:flex;flex-direction:column;flex-shrink:0}
.list{flex:1;overflow-y:auto}
.si{padding:7px 10px;border-bottom:1px solid #f0f0f0;cursor:pointer;font-size:12px;
  display:flex;align-items:center;gap:6px;transition:background .1s}
.si:hover{background:#f5f5f5}
.si.act{background:#e3f2fd;border-left:3px solid #2196F3}
.ft{display:inline-block;padding:1px 5px;border-radius:3px;font-size:9px;font-weight:700;
  color:#fff;min-width:22px;text-align:center;white-space:nowrap;flex-shrink:0}
.ft-not_visible{background:#e74c3c}
.ft-wrong_pick{background:#e67e22}
.ft-pick_then_navigate{background:#f39c12;color:#333!important}
.ft-place_then_pick{background:#27ae60}
.ft-retry_pick{background:#3498db}
.ft-retry_place{background:#9b59b6}
.ft-wrong_location{background:#1abc9c}
.sid{flex:1;font-family:'SF Mono',Menlo,Consolas,monospace;font-size:10px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#555}
.spl{font-size:9px;padding:1px 4px;border-radius:2px;background:#f0f0f0;color:#888;flex-shrink:0}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.dg{background:#27ae60}.db{background:#e74c3c}.du{background:#f39c12}

.ebar{padding:8px 10px;border-top:1px solid #e0e0e0;background:#fafafa}
.ebar button{padding:4px 10px;border:1px solid #ddd;border-radius:3px;background:#fff;
  cursor:pointer;font-size:11px;margin-right:4px}
.ebar button:hover{background:#f0f0f0}
.ebar .st{font-size:10px;color:#888;margin-top:4px}

/* ── Detail ── */
.det{flex:1;overflow-y:auto;padding:20px 24px}
.empty{text-align:center;padding:80px 20px;color:#bbb;font-size:15px}

.nav{display:flex;align-items:center;gap:10px;margin-bottom:16px}
.nav button{padding:5px 14px;border:1px solid #ddd;border-radius:4px;background:#fff;
  cursor:pointer;font-size:12px}
.nav button:hover{background:#f0f0f0}
.nav .pos{font-size:12px;color:#888}

/* ── Images ── */
.imgs{display:flex;gap:16px;margin-bottom:20px}
.ib{flex:1;text-align:center;background:#fafafa;border-radius:8px;padding:8px;border:1px solid #eee}
.ib img{max-width:100%;max-height:340px;border-radius:4px;display:block;margin:0 auto;cursor:zoom-in}
.ib .lb{font-size:11px;color:#999;margin-top:6px}
.noimg{padding:60px 20px;color:#ccc}

/* zoom overlay */
.overlay{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.85);
  display:flex;align-items:center;justify-content:center;z-index:999;cursor:zoom-out}
.overlay img{max-width:95vw;max-height:95vh;border-radius:4px}

/* ── Annotation ── */
.ap{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:14px 16px;margin-bottom:20px}
.ap h3{font-size:13px;margin-bottom:8px;color:#555}
.ab{display:flex;gap:6px;margin-bottom:8px;align-items:center;flex-wrap:wrap}
.ab button{padding:5px 14px;border:2px solid #ddd;border-radius:5px;background:#fff;
  cursor:pointer;font-size:12px;font-weight:600;transition:.15s}
.ab button:hover{filter:brightness(.95)}
.sg{border-color:#27ae60!important;background:#d5f5e3!important}
.sb{border-color:#e74c3c!important;background:#fadbd8!important}
.su{border-color:#f39c12!important;background:#fef9e7!important}
.ab label{font-size:11px;color:#888;margin-left:auto;display:flex;align-items:center;gap:4px}
.ap textarea{width:100%;height:50px;border:1px solid #e0e0e0;border-radius:4px;padding:6px 8px;
  font-size:12px;resize:vertical;font-family:inherit}

/* ── Metadata table ── */
.sec{font-size:14px;font-weight:600;margin:16px 0 8px;color:#444}
.mt{width:100%;border-collapse:collapse}
.mt th{text-align:left;padding:5px 8px;background:#f9f9f9;border:1px solid #eee;
  font-size:11px;color:#888;width:160px;vertical-align:top;font-weight:500}
.mt td{padding:5px 8px;border:1px solid #eee;font-size:12px;word-break:break-word}
.mt td.jc{font-family:'SF Mono',Menlo,Consolas,monospace;font-size:11px;white-space:pre-wrap}

/* quality badge */
.qb{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700;color:#fff}
.q6{background:#27ae60}.q5{background:#2ecc71}.q4{background:#3498db}
.q3{background:#f39c12}.q2{background:#e67e22}.q1{background:#e74c3c}

/* ── Footer ── */
.foot{font-size:10px;color:#aaa;padding:6px 20px;background:#fafafa;border-top:1px solid #eee;
  flex-shrink:0;display:flex;gap:16px}
.foot kbd{background:#e8e8e8;padding:0 4px;border-radius:2px;font-family:monospace;font-size:10px}
</style>
</head>
<body>

<div class="top">
  <h1>🔍 Dataset Inspector</h1>
  <div class="flt" id="flt">
    <span class="lbl">Type:</span>
    <button class="on" data-t="all">All</button>
    <!-- subtype buttons injected by JS -->
    <span class="sep">│</span>
    <span class="lbl">Split:</span>
    <button class="on" data-s="all">All</button>
    <button data-s="train">Train</button>
    <button data-s="valid_seen">Val-Seen</button>
    <button data-s="valid_unseen">Val-Unseen</button>
  </div>
  <input id="q" placeholder="Search sample ID …">
  <span class="cnt" id="cnt"></span>
</div>

<div class="main">
  <div class="side">
    <div class="list" id="lst"></div>
    <div class="ebar">
      <button onclick="doExport()">📥 Export</button>
      <button onclick="doImport()">📤 Import</button>
      <button onclick="doReset()" style="color:#c0392b">🗑 Reset</button>
      <div class="st" id="ast"></div>
    </div>
  </div>
  <div class="det" id="det">
    <div class="empty">← Select a sample from the list to begin</div>
  </div>
</div>

<div class="foot">
  <span><kbd>↑</kbd><kbd>↓</kbd> navigate</span>
  <span><kbd>1</kbd> Good &nbsp;<kbd>2</kbd> Bad &nbsp;<kbd>3</kbd> Unsure</span>
  <span><kbd>Esc</kbd> clear annotation</span>
</div>

<!-- zoom overlay (hidden) -->
<div class="overlay" id="ov" style="display:none" onclick="this.style.display='none'">
  <img id="ovimg">
</div>

<script>
/* ── Data (injected by server) ── */
const ALL = __PAYLOAD__;
const SUBTYPES = __SUBTYPES__;

/* ── Build subtype filter buttons ── */
(function(){
  const flt = document.getElementById('flt');
  // Find the first separator
  const firstSep = flt.querySelector('.sep');
  SUBTYPES.forEach(function(st){
    const b = document.createElement('button');
    b.dataset.t = st;
    // short label
    const shorts = {
      'not_visible':'not_vis','wrong_pick':'wrong_pk','pick_then_navigate':'pk→nav',
      'place_then_pick':'pl→pk','retry_pick':'retry_pk','retry_place':'retry_pl',
      'wrong_location':'wrong_loc'
    };
    b.textContent = shorts[st] || st;
    b.title = st;
    flt.insertBefore(b, firstSep);
  });
})();

/* ── State ── */
let ann = JSON.parse(localStorage.getItem('dsinsp_ann') || '{}');
let fT = 'all', fS = 'all', qStr = '';
let flt = [...ALL], cur = -1, adv = true;

/* ── Helpers ── */
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function saveAnn() { localStorage.setItem('dsinsp_ann', JSON.stringify(ann)); updSt(); }

function updSt() {
  let g=0,b=0,u=0;
  for (const v of Object.values(ann)) {
    if (v.l==='good') g++; else if (v.l==='bad') b++; else if (v.l==='unsure') u++;
  }
  document.getElementById('ast').textContent =
    '\u2713'+g+'  \u2717'+b+'  ?'+u+'  \u2502 '+(g+b+u)+'/'+ALL.length+' reviewed';
}

/* ── Filter & render ── */
function apply() {
  flt = ALL.filter(function(s) {
    if (fT !== 'all' && s.subtype !== fT) return false;
    if (fS !== 'all' && s.split !== fS) return false;
    if (qStr && !s.sample_id.toLowerCase().includes(qStr.toLowerCase())) return false;
    return true;
  });
  cur = flt.length ? 0 : -1;
  rndrList(); rndrDet();
  document.getElementById('cnt').textContent = flt.length + ' / ' + ALL.length + ' samples';
}

function rndrList() {
  const el = document.getElementById('lst');
  let h = '';
  for (let i = 0; i < flt.length; i++) {
    const s = flt[i], a = ann[s.sample_id];
    const dc = a && a.l ? (a.l==='good'?'dg':a.l==='bad'?'db':'du') : '';
    const st = s.subtype || '?';
    h += '<div class="si'+(i===cur?' act':'')+'" onclick="sel('+i+')">' +
      '<span class="ft ft-'+st+'">'+esc(st)+'</span>' +
      '<span class="sid">'+esc(s.sample_id)+'</span>' +
      '<span class="spl">'+esc(s.split)+'</span>' +
      (dc ? '<span class="dot '+dc+'"></span>' : '') +
      '</div>';
  }
  el.innerHTML = h;
  const a = el.querySelector('.act');
  if (a) a.scrollIntoView({block:'nearest'});
}

function sel(i) { cur = i; rndrList(); rndrDet(); }
function go(d) { const n = cur + d; if (n >= 0 && n < flt.length) sel(n); }

/* ── Detail panel ── */
function rndrDet() {
  const el = document.getElementById('det');
  if (cur < 0 || !flt.length) {
    el.innerHTML = '<div class="empty">No sample selected</div>'; return;
  }
  const s = flt[cur], a = ann[s.sample_id] || {};
  const skip = new Set(['_img_before','_img_after']);
  const keys = Object.keys(s).filter(function(k){ return !skip.has(k); });

  // Quality badge
  var qs = s.quality_score;
  var qcls = 'q'+(qs > 6 ? 6 : qs < 1 ? 1 : qs);

  // metadata rows
  let rows = '';
  for (const k of keys) {
    let v = s[k], isO = v !== null && typeof v === 'object';
    let cell;
    if (k === 'quality_score') {
      cell = '<span class="qb '+qcls+'">'+esc(v)+'</span>';
    } else if (k === 'same_frame') {
      cell = v ? '✅ Yes (same frame)' : '❌ No (different frames)';
    } else if (isO) {
      cell = '<span class="jc">'+esc(JSON.stringify(v,null,2))+'</span>';
    } else {
      cell = esc(String(v));
    }
    rows += '<tr><th>'+esc(k)+'</th><td>'+cell+'</td></tr>';
  }

  // images
  const mkImg = function(f, label) {
    if (!f) return '<div class="ib"><div class="noimg">No image</div><div class="lb">'+label+'</div></div>';
    return '<div class="ib"><img src="/images/'+encodeURI(f)+'" onerror="this.outerHTML=\'<div class=noimg>Image not found<br><small>'+esc(f)+'</small></div>\'" onclick="zoom(this.src)"><div class="lb">'+label+'<br><small style=color:#bbb>'+esc(f.split('/').pop())+'</small></div></div>';
  };

  var sfLabel = s.same_frame ? ' <span style="color:#e74c3c;font-size:10px">(⚠ same frame)</span>' : '';

  el.innerHTML =
    '<div class="nav">' +
      '<button onclick="go(-1)">\u25C0 Prev</button>' +
      '<span class="pos">'+(cur+1)+' / '+flt.length+'</span>' +
      '<button onclick="go(1)">Next \u25B6</button>' +
      '<span style="margin-left:12px" class="qb '+qcls+'">Q'+qs+'</span>' +
      '<span class="ft ft-'+s.subtype+'" style="margin-left:6px">'+esc(s.subtype)+'</span>' +
    '</div>' +

    '<div class="imgs">' +
      mkImg(s._img_before, 'Before (failure state)') +
      mkImg(s._img_after,  'After (recovery)' + sfLabel) +
    '</div>' +

    '<div class="ap">' +
      '<h3>Annotation</h3>' +
      '<div class="ab">' +
        '<button class="'+(a.l==='good'?'sg':'')+'" onclick="mark(\'good\')">\u2713 Good (1)</button>' +
        '<button class="'+(a.l==='bad'?'sb':'')+'" onclick="mark(\'bad\')">\u2717 Bad (2)</button>' +
        '<button class="'+(a.l==='unsure'?'su':'')+'" onclick="mark(\'unsure\')">? Unsure (3)</button>' +
        '<label><input type="checkbox" '+(adv?'checked':'')+' onchange="adv=this.checked"> auto-next</label>' +
      '</div>' +
      '<textarea id="nt" placeholder="Notes (optional) …" onblur="saveNt()">'+esc(a.n||'')+'</textarea>' +
    '</div>' +

    '<div class="sec">\uD83D\uDCCB Metadata</div>' +
    '<table class="mt">' + rows + '</table>';
}

/* ── Zoom ── */
function zoom(src) {
  document.getElementById('ovimg').src = src;
  document.getElementById('ov').style.display = 'flex';
}

/* ── Annotation ── */
function mark(l) {
  if (cur < 0) return;
  const sid = flt[cur].sample_id;
  if (!ann[sid]) ann[sid] = {};
  if (ann[sid].l === l) {
    delete ann[sid].l;
    if (!ann[sid].n) delete ann[sid];
  } else {
    ann[sid].l = l;
  }
  saveAnn(); rndrList(); rndrDet();
  if (adv && ann[sid] && ann[sid].l) setTimeout(function(){ go(1); }, 120);
}

function saveNt() {
  if (cur < 0) return;
  const sid = flt[cur].sample_id;
  const v = (document.getElementById('nt') || {}).value || '';
  if (!ann[sid]) ann[sid] = {};
  ann[sid].n = v.trim();
  if (!ann[sid].n && !ann[sid].l) delete ann[sid];
  saveAnn();
}

/* ── Export / Import / Reset ── */
function doExport() {
  const blob = new Blob([JSON.stringify(ann, null, 2)], {type:'application/json'});
  const u = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = u; a.download = 'annotations.json'; a.click();
  URL.revokeObjectURL(u);
}

function doImport() {
  const inp = document.createElement('input'); inp.type = 'file'; inp.accept = '.json';
  inp.onchange = function(e) {
    const r = new FileReader();
    r.onload = function() {
      try {
        const d = JSON.parse(r.result);
        Object.assign(ann, d); saveAnn(); rndrList(); rndrDet();
        alert('Imported ' + Object.keys(d).length + ' annotations');
      } catch(e) { alert('Invalid JSON file'); }
    };
    r.readAsText(e.target.files[0]);
  };
  inp.click();
}

function doReset() {
  if (!confirm('Clear ALL annotations? This cannot be undone.')) return;
  ann = {}; saveAnn(); rndrList(); rndrDet();
}

/* ── Filters ── */
document.getElementById('flt').onclick = function(e) {
  if (e.target.tagName !== 'BUTTON') return;
  const t = e.target.dataset.t, s = e.target.dataset.s;
  if (t !== undefined) {
    fT = t;
    document.querySelectorAll('[data-t]').forEach(function(b){ b.classList.toggle('on', b.dataset.t===t); });
  }
  if (s !== undefined) {
    fS = s;
    document.querySelectorAll('[data-s]').forEach(function(b){ b.classList.toggle('on', b.dataset.s===s); });
  }
  apply();
};

document.getElementById('q').oninput = function(e) { qStr = e.target.value; apply(); };

/* ── Keyboard shortcuts ── */
document.onkeydown = function(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  switch (e.key) {
    case 'ArrowUp':   e.preventDefault(); go(-1); break;
    case 'ArrowDown': e.preventDefault(); go(1);  break;
    case '1': mark('good');   break;
    case '2': mark('bad');    break;
    case '3': mark('unsure'); break;
    case 'Escape':
      if (document.getElementById('ov').style.display !== 'none') {
        document.getElementById('ov').style.display = 'none';
      } else if (cur >= 0) {
        const sid = flt[cur].sample_id;
        delete ann[sid]; saveAnn(); rndrList(); rndrDet();
      }
      break;
  }
};

/* ── Boot ── */
apply(); updSt();
</script>
</body>
</html>""".replace('__PAYLOAD__', JS_DATA).replace('__SUBTYPES__', JS_SUBTYPES)

# ── HTTP Server ─────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/', '/index.html'):
            body = HTML.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith('/images/'):
            rel = urllib.parse.unquote(self.path[8:])
            fp = (IMGS / rel).resolve()
            # security: must be under IMGS
            if not str(fp).startswith(str(IMGS.resolve())) or not fp.is_file():
                self.send_error(404)
                return
            ct = mimetypes.guess_type(str(fp))[0] or 'application/octet-stream'
            data = fp.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', ct)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'public, max-age=3600')
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        pass  # suppress request logs

# ── Main ────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Dataset Inspector')
    ap.add_argument('--port', type=int, default=8888)
    args = ap.parse_args()

    print(f"🌐 Open in browser:  http://localhost:{args.port}")
    print(f"   Press Ctrl+C to stop\n")
    try:
        HTTPServer(('', args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Bye!")