"""Validate recordings, store reviews, and build reproducible training datasets."""
import argparse
import csv
import hashlib
import io
import json
import math
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

LAPTOP_DIR = Path(__file__).resolve().parent.parent
CLASSES = ("LEFT", "RIGHT", "STRAIGHT")
ACTIONS = (*CLASSES, "STOP")
DECISIONS = ("unreviewed", "keep", "exclude")
SOURCE_FIELDS = ("image", "action", "saved_ts", "action_ts", "sha256", "issues")
REVIEW_FIELDS = (*SOURCE_FIELDS, "decision", "label_override", "reason")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def scan_source(source):
    """Index CSV rows, including damaged/missing images for explicit exclusion."""
    source = Path(source).resolve()
    rows = []
    seen = set()
    with (source / "labels.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(SOURCE_FIELDS[:4]).issubset(reader.fieldnames or []):
            raise ValueError("labels.csv must contain image, action, saved_ts, action_ts")
        for raw in reader:
            image = raw["image"]
            if not image or Path(image).name != image or "/" in image or "\\" in image:
                raise ValueError(f"Unsafe image name: {image!r}")
            if image in seen:
                raise ValueError(f"Duplicate CSV image name: {image}")
            seen.add(image)
            row = {key: raw[key] for key in SOURCE_FIELDS[:4]}
            issues = []
            if Path(image).suffix.lower() not in (".jpg", ".jpeg", ".png"):
                issues.append("unsupported_image_type")
            if row["action"] not in ACTIONS:
                issues.append("invalid_label")
            for key in ("saved_ts", "action_ts"):
                try:
                    if not math.isfinite(float(row[key])):
                        raise ValueError()
                except (ValueError, TypeError):
                    issues.append(f"invalid_{key}")
            path = source / "images" / image
            if path.is_symlink() or not path.resolve().is_relative_to(source / "images"):
                raise ValueError(f"Image must stay inside images/: {image}")
            row["sha256"] = ""
            if not path.is_file():
                issues.append("missing_image")
            else:
                data = path.read_bytes()
                row["sha256"] = digest(data)
                decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if decoded is None:
                    issues.append("unreadable_image")
            row["issues"] = "|".join(issues)
            rows.append(row)
    if not rows:
        raise ValueError("Source contains no CSV samples")
    extras = sorted(p.name for p in (source / "images").iterdir()
                    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
                    and p.name not in seen) if (source / "images").exists() else []
    return rows, extras


def validate_decision(row):
    if row["decision"] not in DECISIONS:
        raise ValueError(f"Invalid decision for {row['image']}")
    if row["label_override"] and row["label_override"] not in ACTIONS:
        raise ValueError(f"Invalid label override for {row['image']}")
    if row["issues"] and row["decision"] == "keep":
        raise ValueError(f"Exclude damaged/invalid sample: {row['image']} ({row['issues']})")


def load_review(source, review_path, create=False):
    source_rows, extras = scan_source(source)
    review_path = Path(review_path)
    if not review_path.exists():
        if not create:
            raise FileNotFoundError(f"Review required: {review_path}")
        rows = [dict(row, decision="unreviewed", label_override="", reason="")
                for row in source_rows]
        save_review(review_path, rows)
        return rows, extras
    with review_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(REVIEW_FIELDS):
            raise ValueError(f"Unexpected review schema: {review_path}")
        rows = list(reader)
    if len(rows) != len(source_rows):
        raise ValueError("Source changed since review: row count differs")
    for source_row, row in zip(source_rows, rows):
        if any(source_row[k] != row[k] for k in SOURCE_FIELDS):
            raise ValueError(f"Source changed since review: {source_row['image']}")
        validate_decision(row)
    return rows, extras


def save_review(path, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=REVIEW_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, buffer.getvalue())


def summary(rows, extras):
    kept = Counter(r["label_override"] or r["action"] for r in rows if r["decision"] == "keep")
    return {"samples": len(rows), "decisions": dict(Counter(r["decision"] for r in rows)),
            "labels": dict(Counter(r["label_override"] or r["action"] for r in rows)),
            "kept_labels": {label: kept[label] for label in CLASSES},
            "issues": dict(Counter(issue for r in rows for issue in r["issues"].split("|") if issue)),
            "unreferenced_images": extras}


def read_classes(data_dir, requested=None):
    """Read dataset classes and optionally select an ordered training subset."""
    path = Path(data_dir) / "labels.json"
    manifest = Path(data_dir) / "dataset.json"
    if manifest.exists():
        classes = json.loads(manifest.read_text())["classes"]
    else:
        classes = json.loads(path.read_text()) if path.exists() else list(CLASSES)
    if (not isinstance(classes, list) or len(classes) < 2
            or any(c not in CLASSES for c in classes) or len(set(classes)) != len(classes)):
        raise ValueError("Class metadata must list two or three distinct driving classes in output order")
    if requested is not None:
        if (len(requested) < 2 or len(set(requested)) != len(requested)
                or any(c not in classes for c in requested)):
            raise ValueError(f"Select at least two distinct dataset classes from {classes}")
        return list(requested)
    return classes


def validate_split_classes(data_dir, split, classes):
    for label in classes:
        folder = Path(data_dir) / split / label
        if not any(p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".gif")
                   for p in folder.rglob("*")):
            raise ValueError(f"No samples for selected class: {split}/{label}")


def check_model_classes(model_dir, classes):
    path = Path(model_dir) / "labels.json"
    if path.exists():
        expected = json.loads(path.read_text())
    else:
        labels = Path(model_dir) / "labels.txt"
        expected = [line.split()[-1] for line in labels.read_text().splitlines() if line.strip()] if labels.exists() else list(CLASSES)
    if expected != classes:
        raise ValueError(f"Model classes {expected} differ from dataset classes {classes}")


SPLITS = ("train", "val", "test")


def build_dataset(sources, output, seed=42):
    """Pool reviewed samples, deduplicate, then split by class with a fixed seed."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new version: {output}")
    if not sources:
        raise ValueError("Specify at least one raw session directory")
    classes = list(CLASSES)
    selected, audit, provenance, source_ids = [], Counter(), [], set()
    source_paths = set()
    for source in sorted(Path(path).resolve() for path in sources):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Output must be separate from raw sources")
        source_id = source.name
        if not source_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in source_id):
            raise ValueError("Session names must use letters, digits, underscores or hyphens")
        if source_id in source_ids or source in source_paths:
            raise ValueError("Duplicate session name or path")
        source_ids.add(source_id)
        source_paths.add(source)
        review = source / "review.csv"
        rows, extras = load_review(source, review)
        source_path = source.relative_to(LAPTOP_DIR).as_posix() if source.is_relative_to(LAPTOP_DIR) else source.as_posix()
        provenance.append({"path": source_path, "id": source_id,
                           "review_sha256": digest(review.read_bytes()),
                           "labels_sha256": digest((source / "labels.csv").read_bytes()),
                           "unreferenced_images": extras})
        audit["unreferenced_images"] += len(extras)
        metadata_path = source / "session.json"
        if metadata_path.exists():
            provenance[-1]["session_metadata"] = json.loads(metadata_path.read_text())
        for row in rows:
            audit["input_samples"] += 1
            if row["decision"] == "exclude":
                audit["manually_excluded"] += 1
                continue
            if row["decision"] == "unreviewed":
                audit["unreviewed_skipped"] += 1
                continue
            if row["issues"]:
                raise ValueError(f"Cannot include invalid sample: {source_id}/{row['image']}")
            label = row["label_override"] or row["action"]
            if label not in classes:
                audit["non_driving_samples"] += 1
                continue
            selected.append({**row, "source_id": source_id, "source_path": source_path,
                             "label": label, "file": source / "images" / row["image"]})
    if not selected:
        raise ValueError("No kept driving samples; mark selected images Keep before building")
    groups = defaultdict(list)
    for row in selected:
        groups[row["sha256"]].append(row)
    final = []
    for sha, group in sorted(groups.items()):
        if len({r["label"] for r in group}) != 1:
            raise ValueError(f"Identical image has conflicting labels: {sha}")
        canonical = min(group, key=lambda r: (r["source_id"], r["image"]))
        canonical["duplicate_sources"] = "|".join(
            f"{r['source_id']}/{r['image']}" for r in sorted(group, key=lambda r: (r["source_id"], r["image"])))
        final.append(canonical)
        audit["duplicate_copies_removed"] += len(group) - 1
    for label in classes:
        samples = [r for r in final if r["label"] == label]
        random.Random(f"{seed}:{label}").shuffle(samples)
        count = len(samples)
        val_count = max(1, round(count * 0.15)) if count >= 3 else 0
        test_count = max(1, round(count * 0.15)) if count >= 3 else 0
        train_count = count - val_count - test_count
        for index, row in enumerate(samples):
            row["split"] = "train" if index < train_count else "val" if index < train_count + val_count else "test"
    counts = Counter((r["split"], r["label"]) for r in final)
    empty = [f"{s}/{c}" for s in SPLITS for c in classes if not counts[(s, c)]]
    if any(not any(counts[(s, c)] for c in classes) for s in SPLITS):
        raise ValueError("Not enough reviewed samples to populate train/val/test; review more data")
    report = {"schema_version": 1, "classes": classes, "audit": dict(audit),
              "empty_classes": empty, "seed": seed,
              "split_policy": "pooled, content-deduplicated, stratified by class",
              "target_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
              "counts": {s: {c: counts[(s, c)] for c in classes} for s in SPLITS}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".prepare-", dir=output.parent))
    try:
        for split in SPLITS:
            for label in classes:
                (temporary / split / label).mkdir(parents=True)
        fields = ("source_id", "source_path", "image", "action", "label", "saved_ts", "action_ts",
                  "sha256", "reason", "split", "output_path", "duplicate_sources")
        with (temporary / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in final:
                relative = f"{row['split']}/{row['label']}/{row['source_id']}__{row['image']}"
                if digest(row["file"].read_bytes()) != row["sha256"]:
                    raise ValueError(f"Source changed while building: {row['file']}")
                shutil.copyfile(row["file"], temporary / relative)
                writer.writerow({k: relative if k == "output_path" else row[k] for k in fields})
        write_json(temporary / "dataset.json", {
            **report, "sources": provenance,
            "builder_sha256": digest(Path(__file__).read_bytes()),
        })
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return report


class ReviewStore:
    def __init__(self, source, review=None):
        self.source = Path(source).resolve()
        self.review = Path(review).resolve() if review else self.source / "review.csv"
        if ((self.source == self.review or self.source in self.review.parents)
                and self.review != self.source / "review.csv"):
            raise ValueError("Only review.csv may be written inside a raw source directory")
        self.rows, self.extras = load_review(self.source, self.review, create=True)
        self.undo_stack = []
        self.revision = 0

    def payload(self):
        return {"source": self.source.name, "review": str(self.review), "rows": self.rows,
                "summary": summary(self.rows, self.extras), "revision": self.revision,
                "can_undo": bool(self.undo_stack)}

    def update(self, request):
        if request.get("revision") != self.revision:
            raise ValueError("Review changed in another tab; reload before editing")
        if request.get("undo"):
            if not self.undo_stack:
                raise ValueError("Nothing to undo")
            before = self.undo_stack[-1]
            updated = list(self.rows)
            for index, row in before:
                updated[index] = row
            self.persist(updated)
            self.undo_stack.pop()
        else:
            indices = request.get("indices")
            if (not isinstance(indices, list) or not indices or len(indices) > len(self.rows)
                    or any(type(i) is not int or not 0 <= i < len(self.rows) for i in indices)
                    or len(set(indices)) != len(indices)):
                raise ValueError("Select valid samples")
            changes = request.get("changes", {})
            if not isinstance(changes, dict) or not changes or set(changes) - {"decision", "label_override", "reason"}:
                raise ValueError("Update decision, label_override or reason")
            if any(not isinstance(v, str) or len(v) > 2000 for v in changes.values()):
                raise ValueError("Review fields must be short strings")
            updated = list(self.rows)
            before = [(i, self.rows[i]) for i in indices]
            for index in indices:
                row = {**self.rows[index], **changes}
                validate_decision(row)
                updated[index] = row
            self.persist(updated)
            self.undo_stack.append(before)
            self.undo_stack = self.undo_stack[-30:]
        self.revision += 1
        return self.payload()

    def persist(self, rows):
        # Detect edits made outside this server before replacing the review file.
        current, _ = load_review(self.source, self.review)
        if current != self.rows:
            raise ValueError("Review CSV changed externally; restart the reviewer")
        save_review(self.review, rows)
        self.rows = rows


def review_page(token):
    """Render the embedded browser UI; no separate page file is needed."""
    html = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Line-follow dataset review</title>
<style>
:root{font:14px system-ui;color:#172033;background:#f3f5f8}body{margin:0}header{background:#fff;padding:16px 22px;border-bottom:1px solid #ccd3de}h1{font-size:20px;margin:0 0 8px}button,select,input{font:inherit;padding:6px 9px;border:1px solid #b9c3d0;border-radius:5px;background:white}button{cursor:pointer}button:disabled{opacity:.5;cursor:default}button:hover{background:#e7efff}label{display:inline-block;margin:4px 8px 4px 0}#status{white-space:pre-wrap;color:#334155}#error{color:#a51616;white-space:pre-wrap}main{display:grid;grid-template-columns:1fr 330px;gap:16px;padding:16px}#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(148px,1fr));gap:10px;align-content:start}.card{background:#fff;border:2px solid transparent;border-radius:6px;padding:7px;text-align:left;min-width:0}.card.selected{border-color:#2563eb}.card img{width:100%;height:130px;object-fit:contain;background:#e2e8f0}.card p{margin:4px 0;overflow-wrap:anywhere}.keep{color:#087443}.exclude{color:#aa2424}.unreviewed{color:#775b00}aside{position:sticky;top:12px;align-self:start;background:white;border-radius:8px;padding:14px}aside img{width:100%;background:#e2e8f0}#detail{white-space:pre-wrap;overflow-wrap:anywhere}#saveState{font-weight:600}footer{padding:10px 22px}#reason{width:220px}#info{white-space:pre-wrap}details{margin:10px 0}summary{cursor:pointer}@media(max-width:850px){main{grid-template-columns:1fr}aside{position:static}}
</style>
<header>
<h1 id="title">Dataset review</h1>
<div id="status"></div><div id="classCounts"></div><details><summary>Integrity report and review file</summary><div id="info"></div></details>
<div>
<label>Label <select id="label"><option value="">All labels</option><option>LEFT</option><option>RIGHT</option><option>STRAIGHT</option><option>STOP</option></select></label>
<label>Decision <select id="decision"><option value="">All decisions</option><option value="reviewed">Reviewed (keep + exclude)</option><option>unreviewed</option><option>keep</option><option>exclude</option></select></label>
<label><input id="issues" type="checkbox">Only issues</label>
<button id="selectPage">Select page</button><button id="selectFiltered">Select all filtered</button><button id="clear">Clear selection</button>
</div>
<div>
<button id="keep">Keep (K)</button><button id="exclude">Exclude (X)</button><button id="reset">Unreviewed (U)</button>
<label>Training label <select id="override"><option value="">Original label</option><option>LEFT</option><option>RIGHT</option><option>STRAIGHT</option><option>STOP</option></select></label><button id="applyLabel">Apply label</button>
<label>Reason <input id="reason" placeholder="Optional, e.g. lane_out_of_view"></label><button id="applyReason">Apply reason</button><button id="undo">Undo</button>
</div>
<div id="saveState">Loading…</div><div id="error" role="alert"></div>
</header>
<main><div><div id="pager"><button id="prevPage">Previous page</button> <span id="pageText"></span> <button id="nextPage">Next page</button></div><div id="grid"></div></div>
<aside><strong>Selected frame</strong><img id="preview" alt="Select a frame to inspect it"><div id="detail"></div><p><button id="prevFrame">← Previous</button> <button id="nextFrame">Next →</button></p><p>Click: select one. Ctrl/Cmd-click: toggle. Shift-click: select a range in the current filter. Arrow keys inspect adjacent frames in recording order.</p><p>Decisions save automatically. Original images and labels remain unchanged. STOP is excluded when building. Choose driving classes later at training time.</p></aside></main>
<footer>Review one source at a time. Change source by restarting the reviewer with another --source.</footer>
<script>
const token='__TOKEN__', $=id=>document.getElementById(id), size=60;
let state, filtered=[], selected=new Set(), page=0, anchor=null, focus=null, busy=false;
function effective(row){return row.label_override||row.action}
function error(e){$('error').textContent=e.message||String(e)}
function matchesDecision(row,decision){return !decision||(decision==='reviewed'?(row.decision==='keep'||row.decision==='exclude'):row.decision===decision)}
function refilter(){filtered=state.rows.map((r,i)=>i).filter(i=>{const r=state.rows[i];return (!$('label').value||effective(r)===$('label').value)&&matchesDecision(r,$('decision').value)&&(!$('issues').checked||r.issues)});page=0;selected.clear();anchor=null;render()}
function inspect(i){focus=i;const r=state.rows[i];$('preview').src='/image/'+i;$('detail').textContent=`${r.image}\nOriginal: ${r.action}\nTraining: ${effective(r)}\nDecision: ${r.decision}\nSaved: ${r.saved_ts}\nKey event: ${r.action_ts}\nIssues: ${r.issues||'none'}\nReason: ${r.reason||'none'}`}
function choose(i,e){if(e.shiftKey&&anchor!==null&&filtered.includes(anchor)){const a=filtered.indexOf(anchor),b=filtered.indexOf(i);selected=new Set(filtered.slice(Math.min(a,b),Math.max(a,b)+1))}else if(e.metaKey||e.ctrlKey){selected.has(i)?selected.delete(i):selected.add(i);anchor=i}else{selected=new Set([i]);anchor=i}inspect(i);render()}
function render(){
 const stats=state.summary;$('title').textContent='Dataset review · '+state.source;
 $('status').textContent=`${stats.samples} samples · kept ${stats.decisions.keep||0} · excluded ${stats.decisions.exclude||0} · unreviewed ${stats.decisions.unreviewed||0}`;
 $('classCounts').textContent=`Kept driving samples · STRAIGHT ${stats.kept_labels.STRAIGHT} · RIGHT ${stats.kept_labels.RIGHT} · LEFT ${stats.kept_labels.LEFT}`;
 $('info').textContent=state.review+'\n'+JSON.stringify({issues:stats.issues,unreferenced_images:stats.unreferenced_images},null,2);
 $('saveState').textContent=busy?'Saving…':`${selected.size} selected · saved`;
 $('undo').disabled=busy||!state.can_undo;
 for(const id of ['keep','exclude','reset','applyLabel','applyReason'])$(id).disabled=busy||!selected.size;
 const pages=Math.max(1,Math.ceil(filtered.length/size));page=Math.min(page,pages-1);$('pageText').textContent=`Page ${page+1}/${pages} · ${filtered.length} filtered`;$('prevPage').disabled=page===0;$('nextPage').disabled=page===pages-1;
 $('grid').replaceChildren();for(const i of filtered.slice(page*size,(page+1)*size)){
  const r=state.rows[i],card=document.createElement('button');card.className='card'+(selected.has(i)?' selected':'');card.onclick=e=>{if(!busy)choose(i,e)};
  const img=document.createElement('img');img.loading='lazy';img.src='/image/'+i+'?thumb';img.alt=r.issues||r.image;card.append(img);
  for(const text of [r.image,r.action+(r.label_override?' → '+r.label_override:''),r.decision+(r.issues?' · '+r.issues:'')]){const p=document.createElement('p');p.textContent=text;p.className=r.decision;card.append(p)}$('grid').append(card)
 }
 if(focus!==null)inspect(focus)
}
async function update(changes,undo=false){if(busy||(!undo&&!selected.size))return;busy=true;$('error').textContent='';render();try{
 const res=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify({revision:state.revision,indices:[...selected],changes,undo})});const result=await res.json();if(!res.ok)throw Error(result.error);state=result;
 const oldSelection=new Set(selected),oldPage=page;refilter();selected=new Set([...oldSelection].filter(i=>filtered.includes(i)));page=oldPage;
 }catch(e){error(e)}finally{busy=false;render()}}
for(const id of ['label','decision','issues'])$(id).onchange=()=>{if(!busy)refilter()};
$('selectPage').onclick=()=>{selected=new Set(filtered.slice(page*size,(page+1)*size));render()};
$('selectFiltered').onclick=()=>{selected=new Set(filtered);render()};$('clear').onclick=()=>{selected.clear();render()};
for(const [id,value] of [['keep','keep'],['exclude','exclude'],['reset','unreviewed']])$(id).onclick=()=>update({decision:value,...($('reason').value?{reason:$('reason').value}:{})});
$('applyLabel').onclick=()=>update({label_override:$('override').value});$('applyReason').onclick=()=>update({reason:$('reason').value});$('undo').onclick=()=>update({},true);
$('prevPage').onclick=()=>{page--;render()};$('nextPage').onclick=()=>{page++;render()};
function step(delta){if(busy)return;const i=Math.max(0,Math.min(state.rows.length-1,(focus??0)+delta));selected=new Set([i]);anchor=i;inspect(i);const n=filtered.indexOf(i);if(n>=0)page=Math.floor(n/size);render()}
$('prevFrame').onclick=()=>step(-1);$('nextFrame').onclick=()=>step(1);
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)||busy||!state)return;const k=e.key.toLowerCase();if(k==='arrowleft'||k==='arrowright'){e.preventDefault();step(k==='arrowleft'?-1:1)}else if(['k','x','u'].includes(k)){e.preventDefault();$(k==='k'?'keep':k==='x'?'exclude':'reset').click()}});
fetch('/api/state').then(r=>r.json()).then(s=>{state=s;refilter();$('saveState').textContent='Ready · decisions save automatically'}).catch(error);
</script>
</html>
"""
    return html.replace("__TOKEN__", token).encode("utf-8")


def make_handler(store, token):
    from http.server import BaseHTTPRequestHandler
    from urllib.parse import urlparse

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == "/":
                    self.send(200, review_page(token), "text/html; charset=utf-8")
                elif path == "/api/state":
                    self.send(200, json.dumps(store.payload()).encode())
                elif path.startswith("/image/"):
                    index = int(path.removeprefix("/image/"))
                    if not 0 <= index < len(store.rows):
                        raise ValueError("Invalid image index")
                    row = store.rows[index]
                    data = (store.source / "images" / row["image"]).read_bytes()
                    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("Image cannot be decoded")
                    if urlparse(self.path).query == "thumb":
                        height, width = image.shape[:2]
                        scale = min(1, 160 / max(height, width))
                        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
                    ok, encoded = cv2.imencode(".jpg", image)
                    if not ok:
                        raise ValueError("Image cannot be encoded")
                    self.send(200, encoded.tobytes(), "image/jpeg")
                else:
                    self.send(404, b'{"error":"Not found"}')
            except (ValueError, OSError) as error:
                self.send(400, json.dumps({"error": str(error)}).encode())

        def do_POST(self):
            if self.path != "/api/review" or self.headers.get("X-Review-Token") != token:
                self.send(403, b'{"error":"Invalid review request"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("Expected a JSON object")
                result = store.update(request)
                self.send(200, json.dumps(result).encode())
            except (ValueError, OSError, TypeError, KeyError) as error:
                self.send(400, json.dumps({"error": str(error)}).encode())

        def log_message(self, *args):
            pass

    return Handler


def run_review(source, port=8765):
    import secrets
    from http.server import HTTPServer

    store = ReviewStore(source)
    review = store.review
    server = HTTPServer(("127.0.0.1", port), make_handler(store, secrets.token_hex(24)))
    print(f"Review: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"Decisions: {review}\nCtrl+C stops the server; saved decisions remain.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    review = commands.add_parser("review", help="Review one raw recording in a local browser")
    review.add_argument("--source", required=True, type=Path, help="Directory with labels.csv and images/")
    review.add_argument("--port", type=int, default=8765)
    check = commands.add_parser("check", help="Report source integrity without writing files")
    check.add_argument("--source", required=True, type=Path)
    build = commands.add_parser("build", help="Build a dataset from saved review decisions")
    build.add_argument("--source", required=True, nargs="+", type=Path, help="Reviewed raw session directories")
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "review":
        run_review(args.source, port=args.port)
    elif args.command == "check":
        rows, extras = scan_source(args.source)
        report = summary([dict(r, decision="unreviewed", label_override="") for r in rows], extras)
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps(build_dataset(args.source, args.output, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
