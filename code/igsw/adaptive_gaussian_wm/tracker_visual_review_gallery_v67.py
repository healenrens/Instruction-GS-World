"""Self-contained local gallery, query picker, and human review export."""

from __future__ import annotations

import html
import json
from pathlib import Path


def write_gallery(out, cases, results, revision):
    out = Path(out)
    blocks = []
    for case in cases:
        case_id = case["case_id"]
        pairs = [row for row in results if row["case_id"] == case_id]
        if "point_count" not in case:
            continue
        esc = html.escape
        rows = []
        for row in pairs:
            path = esc(row["directory"])
            key = esc(f"{case_id}/{row['step_ms']}")
            display = row.get("display_filter")
            display_note = (
                f"<p>Showing {display['shown_point_count']}/{display['raw_point_count']} points; "
                f"native visible-position span threshold = {display['threshold_px']:.1f} px. "
                f'Image motion only, not object/accuracy labels. <a href="{path}/motion_filter.json">All selection decisions</a></p>'
                if display
                else ""
            )
            grounded = row.get("grounded_selection")
            if grounded:
                role_note = (
                    "Roles combine native tracks with robot masks at multiple anchors. "
                    "They are uncalibrated hypotheses, not verified identities. "
                    f"Uncertain moving candidates: {grounded['uncertain_motion_candidate_count']}. "
                    if grounded.get("contract")
                    in (
                        "grounded_object_motion_selection_v2",
                        "grounded_object_motion_selection_v3",
                    )
                    else "Query-frame pseudo roles, not verified object labels. "
                )
                display_note = (
                    f"<p>Object-motion targets: {grounded['object_motion_target_count']}; "
                    f"context points: {grounded['context_count']}; total raw queries: {grounded['raw_point_count']}. "
                    "Green = object-motion candidate; orange = robot; purple = unknown; blue = scene. "
                    f"{role_note}Main display uses exactly the exported selection.</p>"
                    f'<p><a href="{path}/motion_filter.json">Per-point selection and thresholds</a> · '
                    f'<a href="{path}/training_candidates.pt">Training candidate export</a></p>'
                    f'<details><summary>All tracked points, BEFORE motion filtering</summary><video controls preload="none" src="{path}/all_queries.mp4"></video></details>'
                )
                if "motion_top_fraction" in grounded:
                    before = grounded["object_motion_candidate_count_before_topk"]
                    retained = grounded["object_motion_target_count"]
                    fraction = grounded["motion_top_fraction"]
                    display_note += (
                        f"<h3>物体运动候选 Top-{fraction:.0%}：{before} → {retained}</h3>"
                        "<p>按每段视频内的有效轨迹运动跨度排名，不是追踪置信度。机械臂、scene、unknown 不参与竞争；"
                        "下方两图仅显示物体候选，上下文仍保留在数据和完整视频中。</p>"
                        f'<div class="videos"><figure><figcaption>筛选前：所有通过运动门槛的物体候选 ({before})</figcaption>'
                        f'<video controls loop preload="none" src="{path}/object_before_topk.mp4"></video></figure>'
                        f"<figure><figcaption>筛选后：导出的物体运动目标 ({retained})</figcaption>"
                        f'<video controls loop preload="none" src="{path}/object_after_topk.mp4"></video></figure></div>'
                    )
            snapshot = (
                f'<details><summary>Full-clip trajectory image</summary><figure><img loading="lazy" src="{path}/trajectories.png" alt="Full visible trajectory segments for displayed points"></figure></details>'
                if "trajectories" in row.get("media", {})
                else ""
            )
            rows.append(f"""
<div class="pair"><h3>{row["step_ms"]} ms request / {row["actual_step_ms"]:.1f} ms actual</h3>
{display_note}
<div class="videos">
<figure><figcaption>Continuous frames</figcaption><video controls loop preload="none" src="{path}/native.mp4"></video></figure>
<figure><figcaption>{row["sampled_frame_count"]} sampled frames, {row["clip_seconds"]:.2f} seconds</figcaption><video controls loop preload="none" src="{path}/sampled.mp4"></video></figure>
</div>
<details><summary>Same timestamps, side by side</summary><video class="wide" controls loop preload="none" src="{path}/comparison.mp4"></video></details>
{snapshot}
<p><a href="{path}/point_rows.csv">All original per-point measurements</a> · <a href="{path}/tracks.pt">All unfiltered tracks</a> · <a href="{path}/result.json">Run metadata</a></p>
<div class="review" data-review="{key}"><label>Point tracking <select data-field="tracking"><option>unreviewed</option><option>stable</option><option>drift</option><option>wrong_surface</option><option>mixed</option><option>unclear</option></select></label>
<label>Occlusion/reappearance <select data-field="occlusion"><option>unreviewed</option><option>not_present</option><option>correct_reappearance</option><option>wrong_reappearance</option><option>visibility_error</option><option>unclear</option></select></label>
<label>Sampling comparison <select data-field="sampling"><option>unreviewed</option><option>similar</option><option>native_better</option><option>sampled_better</option><option>both_bad</option><option>unclear</option></select></label>
<label class="notes">Frame / point IDs / observation <input data-field="notes" type="text"></label></div></div>""")
        proposals = "".join(
            f"<figure><figcaption>frame {view['frame']}: {view['point_count']} queries, mask covers {view['mask_fraction']:.1%}</figcaption>"
            f'<img loading="lazy" src="{esc(case_id)}/{esc(view["overlay"])}" alt="Motion mask and actual tracker query points"></figure>'
            for view in case.get("motion_views", [])
        )
        empty = (
            "<p>No motion queries: empty proposals are retained, not replaced by grid points.</p>"
            if not case["point_count"]
            else ""
        )
        mask_open = "" if any("display_filter" in row for row in pairs) else " open"
        mask_title = (
            "Grounded-SAM-2 proposals: orange is a single-frame robot hypothesis, not a confirmed role"
            if case.get("sampling_kind", "").startswith("grounding_dino_sam2_")
            else "Original proposal masks and queries, BEFORE track filtering (not object labels)"
        )
        provenance = case.get("source_review")
        source_note = ""
        if provenance:
            role_links = (
                f' · <a href="{esc(case_id)}/role_resolution.json">Per-point cross-anchor role evidence</a>'
                f' · <a href="{esc(case_id)}/resolved_queries.pt">Resolved queries</a>'
                if "resolved_role_counts" in case
                else ""
            )
            source_note = (
                f"<p>Indexed file: <code>{esc(provenance['indexed_record']['path'])}</code><br>"
                f"Episode file frames: {provenance['episode_file_frame_range']}; "
                f"tracked file frames: {provenance['selected_file_frames']}. "
                "Original/unedited provenance is NOT established.</p>"
                f'<h3>整段 episode 概览与文件映射</h3><p><a href="{esc(case_id)}/episode_overview.png">直接打开整段概览 PNG</a> · '
                f'<a href="{esc(case_id)}/source_review.json">File and episode mapping / 文件映射 JSON</a>{role_links}</p>'
                "<details open><summary>Whole indexed episode overview / 整段概览（非原版认证）</summary>"
                f'<figure><img loading="lazy" src="{esc(case_id)}/episode_overview.png" alt="Frames sampled across the entire indexed episode, annotated with frame numbers and times"></figure></details>'
                f'<div class="review" data-review="{esc(case_id)}/source"><label>Indexed episode <select data-field="provenance"><option>unreviewed</option><option>continuous_episode</option><option>edited_or_cut</option><option>wrong_episode_range</option><option>wrong_camera</option><option>unclear</option></select></label><label class="notes">Cuts / timestamps / correct path if known <input data-field="notes"></label></div>'
            )
        elif case.get("sampling_kind", "").startswith("grounding_dino_sam2_"):
            source_note = "<p>此份结果没有 episode 概览或文件映射记录；不能用十秒追踪视频代替整段来源检查。</p>"
        blocks.append(f"""
<section data-source="{esc(case["source"])}"><h2>{esc(case_id)}</h2>
<p>{esc(case["group"])} · {case["width"]} × {case["height"]} · {case["record"]["fps"]:g} Hz · {case["clip_seconds"]:.2f} seconds · {esc(case["camera"])} · {case["point_count"]} queries</p>
{source_note}
<details><summary>Consecutive selected frames from indexed video</summary><video class="wide" controls loop preload="none" src="{esc(case_id)}/source.mp4"></video></details>
<details{mask_open}><summary>{mask_title}</summary><div class="videos">{proposals}</div></details>{empty}
<p><a href="{esc(case_id)}/sampling.json">Mask method, thresholds and sampling metadata</a> · <a href="{esc(case_id)}/queries.pt">Query positions and times</a></p>
<div class="review" data-review="{esc(case_id)}/masks"><label>Region selection <select data-field="mask"><option>unreviewed</option><option>moving_object_covered</option><option>oversized_region_background_spill</option><option>small_object_missed</option><option>object_mislabelled_robot</option><option>robot_mislabelled_object</option><option>mostly_arm</option><option>camera_motion</option><option>shadow_or_noise</option><option>empty</option><option>mixed</option></select></label><label class="notes">Incorrect regions / frames <input data-field="notes"></label></div>
<details class="picker" data-case="{esc(case_id)}"><summary>Select points for a manual rerun</summary>
<div class="toolbar"><label>Point group <input class="point-label" value="target"></label><button class="undo" type="button">Undo point</button><button class="clear" type="button">Clear points</button><span class="point-count">0 points</span></div>
<canvas data-image="{esc(case_id)}/anchor.png"></canvas><pre class="point-list"></pre></details>
{"".join(rows)}</section>""")
    payload = json.dumps({"cases": cases, "source_revision": revision}).replace(
        "<", "\\u003c"
    )
    sources = "".join(
        f"<option>{html.escape(source)}</option>"
        for source in sorted({case["source"] for case in cases})
    )
    document = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Point tracker visual review</title><style>
*{box-sizing:border-box}body{margin:0;font:14px system-ui,sans-serif;color:#202522;background:#fff;letter-spacing:0}
header,main{max-width:1500px;margin:auto;padding:20px}header{border-bottom:1px solid #cbd1cc}h1{font-size:24px;margin:0 0 12px}h2{font-size:18px;overflow-wrap:anywhere}h3{font-size:15px}p{color:#4b5550;overflow-wrap:anywhere}a{color:#176147}
section{padding:16px 0 24px;border-bottom:2px solid #aebbb2}.pair{border-top:1px solid #dce2de;padding:8px 0 20px}.videos{display:grid;grid-template-columns:1fr 1fr;gap:12px}
figure{margin:0;min-width:0}figcaption{margin:6px 0}video{width:100%;background:#151817;max-height:640px}.wide{max-width:1200px}.toolbar,.review{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
figure img{display:block;width:100%;height:auto}
button,input,select{font:inherit;padding:6px;border:1px solid #aab6ad;border-radius:4px}button{background:#f0f5f1;cursor:pointer}label{display:flex;gap:6px;align-items:center}.notes{flex:1;min-width:240px}.notes input{min-width:0;flex:1}
details{margin:12px 0}summary{cursor:pointer;padding:6px 0}canvas{display:block;max-width:100%;height:auto;cursor:crosshair;margin-top:10px;background:#202522}pre{white-space:pre-wrap;overflow-wrap:anywhere}
@media(max-width:720px){.videos{grid-template-columns:1fr}header,main{padding:12px}.review label{width:100%;flex-wrap:wrap}h1{font-size:21px}}</style>
<header><h1>Long-clip motion-mask tracker review</h1><p>Motion masks select queries at multiple times, not object labels. Orange = raw proposal; cyan = query before filtering. Stable trajectory color = original point ID. Lines join observed visible, in-frame positions; gaps are not bridged. Filled point = predicted visible; ring = predicted not visible. Query locations are supplied. No independent accuracy labels yet.</p>
<div class="toolbar"><label>Source <select id="source"><option>all</option>__SOURCES__</select></label><button id="export-review">Export observations</button><button id="export-points">Export manual queries</button></div></header>
<main>__BLOCKS__</main><script id="manifest" type="application/json">__PAYLOAD__</script><script>
const manifest=JSON.parse(document.getElementById('manifest').textContent);
const storageKey='tracker-review:'+location.pathname;
const saved=JSON.parse(localStorage.getItem(storageKey)||'{"points":{},"reviews":{}}');
function persist(){localStorage.setItem(storageKey,JSON.stringify(saved));}
function download(name,data){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download=name;a.click();URL.revokeObjectURL(a.href);}
document.getElementById('source').onchange=e=>document.querySelectorAll('section').forEach(s=>s.hidden=e.target.value!=='all'&&s.dataset.source!==e.target.value);
document.getElementById('export-review').onclick=()=>download('human_observations.json',{source_revision:manifest.source_revision,reviews:saved.reviews,interpretation:'human visual review, not dense point ground truth'});
document.getElementById('export-points').onclick=()=>download('manual_queries.json',{...manifest,points:saved.points});
document.querySelectorAll('[data-review]').forEach(row=>row.querySelectorAll('[data-field]').forEach(input=>{
const key=row.dataset.review,field=input.dataset.field;if(saved.reviews[key]?.[field]!==undefined)input.value=saved.reviews[key][field];
input.onchange=()=>{saved.reviews[key]||={};saved.reviews[key][field]=input.value;persist();};}));
document.querySelectorAll('.picker').forEach(picker=>{
const id=picker.dataset.case,canvas=picker.querySelector('canvas'),ctx=canvas.getContext('2d'),img=new Image();
saved.points[id]||=[];
function redraw(){ctx.drawImage(img,0,0);saved.points[id].forEach((p,i)=>{ctx.beginPath();ctx.arc(p.x*(canvas.width-1),p.y*(canvas.height-1),5,0,2*Math.PI);ctx.fillStyle='#ffea00';ctx.fill();ctx.fillStyle='#000';ctx.fillText(i, p.x*(canvas.width-1)+7,p.y*(canvas.height-1));});picker.querySelector('.point-count').textContent=saved.points[id].length+' points';picker.querySelector('.point-list').textContent=saved.points[id].map((p,i)=>i+': '+p.label+' ('+p.x.toFixed(4)+', '+p.y.toFixed(4)+')').join('\n');}
img.onload=()=>{canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;redraw();};
picker.addEventListener('toggle',()=>{if(picker.open&&!img.src)img.src=canvas.dataset.image;});
canvas.onclick=e=>{const box=canvas.getBoundingClientRect();saved.points[id].push({x:Math.max(0,Math.min(1,(e.clientX-box.left)/box.width)),y:Math.max(0,Math.min(1,(e.clientY-box.top)/box.height)),label:picker.querySelector('.point-label').value});persist();redraw();};
picker.querySelector('.undo').onclick=()=>{saved.points[id].pop();persist();redraw();};picker.querySelector('.clear').onclick=()=>{saved.points[id]=[];persist();redraw();};});
</script></html>"""
    document = (
        document.replace("__SOURCES__", sources)
        .replace("__BLOCKS__", "".join(blocks))
        .replace("__PAYLOAD__", payload)
    )
    if any(
        case.get("sampling_kind", "").startswith("grounding_dino_sam2_")
        for case in cases
    ):
        document = document.replace(
            "Long-clip motion-mask tracker review",
            "Grounded object / robot tracker review",
        )
        document = document.replace(
            "Motion masks select queries at multiple times, not object labels. Orange = raw proposal; cyan = query before filtering. Stable trajectory color = original point ID.",
            "SAM masks select queries at multiple times. Green = object candidate, orange = robot context, purple = unknown/contact overlap, blue = scene context. These are query-frame pseudo roles, not per-frame verified identities. Main videos use the exported training selection; all-query videos retain filtered points.",
        )
    if any(
        case.get("sampling_kind") == "grounding_dino_sam2_coverage_queries_v2"
        for case in cases
    ):
        document = document.replace(
            "SAM masks select queries at multiple times. Green = object candidate, orange = robot context, purple = unknown/contact overlap, blue = scene context. These are query-frame pseudo roles, not per-frame verified identities.",
            "SAM masks sample multiple support scales. In proposal images orange is a single-frame robot hypothesis. In tracked videos: green = object candidate, orange = repeated robot-core evidence, purple = unresolved role, blue = scene candidate. Track roles use multiple anchor masks; colors are NOT object IDs or verified labels. Uniform whole-episode overviews are for source inspection only.",
        )
    (out / "index.html").write_text(document, encoding="utf-8")
