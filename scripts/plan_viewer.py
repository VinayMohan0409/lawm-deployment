"""One HTML page per run: what the model predicted and what the robot did, replan by replan.

3D view per replan (layers can be toggled in the legend):
- scene: the camera points;
- hand now: FK at the measured joints, the model's input;
- predicted 1..5: the model's hand at each predicted step (the plan's cumulative delta from the observation it was planned on);
- fitted: the Sharpa hand fitted to the predicted step that is executed (palm pose + 22 angles);
- commanded: after the safety caps (ori_alpha rotation smoothing, finger step cap, translation cap);
- measured after: FK at the joints measured after the move;
- goal (and lift goal).

Charts per replan:
- distance to the goal;
- finger shape progress towards the goal in the palm frame (prediction / fit / command / measured);
- palm rotation error to the goal;
- the predicted hand's scale at steps 1, 3 and 5 (a Sharpa hand cannot change size; < 1 means the model shrinks it).

PYTHONPATH=<lawm>/src:<lawm-deployment> python scripts/plan_viewer.py RUN_DIR [--out page.html] [--points 600]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lawm_deploy import config
from lawm_deploy.hand import Hand
from lawm_deploy.robot_model import RobotModel

PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"


def kabsch(a, b):
    """R, t with R a + t ~ b."""
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    r = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    return r, cb - r @ ca


def scale(a, b):
    """Scale of the best similarity transform from a to b."""
    a0, b0 = a - a.mean(0), b - b.mean(0)
    u, s, vt = np.linalg.svd(a0.T @ b0)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return float((s[0] + s[1] + d * s[2]) / (a0 ** 2).sum())


def load(path):
    return np.load(path, allow_pickle=False) if path.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out")
    ap.add_argument("--points", type=int, default=600, help="hand points drawn per layer")
    ap.add_argument("--scene-points", type=int, default=2500)
    args = ap.parse_args()
    run = Path(args.run)
    meta = json.loads((run / "run.json").read_text())
    cfg = meta["config"]
    robot = RobotModel(config.mount_pose(cfg), seed=cfg["seed"])
    hand = Hand(robot.model, cfg["perception"]["n_hand"], cfg["seed"])
    k = hand.kinematics
    palm = k.point_body == k.root
    fingers = ~palm
    records = [json.loads(line) for line in (run / "records.jsonl").read_text().splitlines()]
    goals = [load(run / "goal.npz")] + ([load(run / "goal_lift.npz")] if (run / "goal_lift.npz").exists() else [])
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(cfg["perception"]["n_hand"], min(args.points, cfg["perception"]["n_hand"]), replace=False))
    names = robot.hand_joint_names
    pip = [i for i, n in enumerate(names) if n.endswith("_PIP")]

    def palm_local(points):  # hand points in the palm's own frame (rotation and position removed)
        r, t = kabsch(k.local[palm], points[palm])
        return (points - t) @ r

    def fk(root, q):
        return hand.points(root, q)

    goal_local = [palm_local(g["coords"][g["segment"] == 0]) for g in goals]
    goal_rot = []
    for g in goals:
        robot.set(g["q_arm"], g["q_hand"])
        goal_rot.append(robot.root_pose()[:3, :3])

    def shape_err(points, stage):  # mean finger-point distance (mm) to the goal hand's finger shape, in the palm frame
        return float(np.linalg.norm(palm_local(points)[fingers] - goal_local[stage][fingers], axis=1).mean() * 1000)

    def rot_err(points, stage):
        r, _ = kabsch(k.local[palm], points[palm])
        return float(np.degrees(Rotation.from_matrix(r @ goal_rot[stage].T).magnitude()))

    frames, chart = [], {key: [] for key in ("replan", "distance", "shape_now", "shape_pred", "shape_fit", "shape_cmd", "shape_meas",
                                              "rot_now", "rot_pred", "rot_fit", "rot_cmd", "rot_meas", "scale1", "scale3", "scale5", "pip_mean")}
    to_mm = lambda p: np.round(np.asarray(p) * 1000).astype(int).tolist()  # noqa: E731
    for rec in records:
        i, stage = rec["replan"], rec.get("stage", 0)
        inp, pred, cmd, after = (load(run / f"{n}_replan_{i:03d}.npz") for n in ("input", "prediction", "command", "after"))
        if inp is None or pred is None:
            continue
        h = inp["segment"] == 0
        now = inp["coords"][h]
        plan_meta = json.loads(str(pred["plan_metadata_json"]))
        source = load(run / plan_meta["delta_reference_archive"]) if plan_meta.get("delta_reference_archive") else inp
        base = source["coords"][source["segment"] == 0]
        steps = [base + pred["delta"][j][source["segment"] == 0] for j in range(len(pred["delta"]))]
        executed = pred["hand_target"]
        layers = dict(now=now, executed=executed, **{f"pred{j + 1}": s for j, s in enumerate(steps)})
        if cmd is not None:
            layers["fitted"] = fk(np.block([[cmd["fitted_root_rotation"], cmd["fitted_root_position"][:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]), cmd["fitted_fingers"])
            layers["commanded"] = fk(cmd["target"] @ robot.root_in_flange, cmd["fingers"])
        if after is not None and rec.get("moved"):
            layers["measured"] = after["coords"][after["segment"] == 0]
        scene = inp["coords"][~h]
        scene = scene[np.sort(rng.choice(len(scene), min(args.scene_points, len(scene)), replace=False))]
        frames.append(dict(replan=i, stage=stage, step=plan_meta.get("predicted_step"), reused=plan_meta.get("reused_plan"),
                           hold=rec.get("hold"), distance=round(rec["distance_mm"], 1), scene=to_mm(scene),
                           layers={n: to_mm(p[pick]) for n, p in layers.items()}))
        chart["replan"].append(i)
        chart["distance"].append(rec["distance_mm"])
        for key, pts in (("now", now), ("pred", executed), ("fit", layers.get("fitted")), ("cmd", layers.get("commanded")), ("meas", layers.get("measured"))):
            chart[f"shape_{key}"].append(None if pts is None else shape_err(pts, stage))
            chart[f"rot_{key}"].append(None if pts is None else rot_err(pts, stage))
        for j in (1, 3, 5):
            chart[f"scale{j}"].append(None if plan_meta.get("reused_plan") else scale(base, steps[j - 1]))
        q = after["q_hand"] if after is not None and rec.get("moved") else inp["q_hand"]
        chart["pip_mean"].append(float(np.degrees(q[pip]).mean()))
    data = dict(title=f"{meta['task']} · {run.name}", frames=frames, chart=chart,
                goals=[to_mm(g["coords"][g["segment"] == 0][pick]) for g in goals],
                goal_pip=float(np.degrees(goals[-1]["q_hand"][pip]).mean()))
    out = Path(args.out or run / "plan_viewer.html")
    out.write_text(PAGE.replace("__PLOTLY__", PLOTLY).replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__TITLE__", data["title"]))
    print(out)


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plan viewer</title>
<script src="__PLOTLY__"></script>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--card:#ffffff;--line:#e7e5e4}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--card:#292524;--line:#44403c}}
:root[data-theme="dark"]{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--card:#292524;--line:#44403c}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1400px;margin:0 auto;padding:12px 16px}
h1{font-size:18px;margin:4px 0 2px} .muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:12px}
@media (max-width:900px){.grid{grid-template-columns:minmax(0,1fr)}}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}
#view{height:620px} .chart{height:200px}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:8px 0}
input[type=range]{flex:1;min-width:200px}
button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
</style></head><body><main>
<h1>__TITLE__</h1>
<div class="muted">Each replan: the model's predicted hand (steps 1–5), the Sharpa hand fitted to the executed step, the command after the safety caps, and the hand measured after the move. Click legend entries to hide layers.</div>
<div class="controls"><button id="prev">◀</button><input id="slider" type="range" min="0" value="0"><button id="next">▶</button><span id="label"></span></div>
<div class="grid"><div class="card"><div id="view"></div></div>
<div><div class="card"><div id="c_dist" class="chart"></div></div><div class="card"><div id="c_shape" class="chart"></div></div>
<div class="card"><div id="c_rot" class="chart"></div></div><div class="card"><div id="c_scale" class="chart"></div></div></div></div>
</main><script>
const D=__DATA__;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const STYLE={scene:['#a8a29e',1.6,'scene'],now:['#2563eb',3,'hand now'],pred1:['#fdba74',2.5,'predicted 1'],pred2:['#fb923c',2.5,'predicted 2'],pred3:['#f97316',2.5,'predicted 3'],
 pred4:['#ea580c',2.5,'predicted 4'],pred5:['#9a3412',2.5,'predicted 5'],executed:['#facc15',3,'executed step target'],fitted:['#a855f7',3,'fitted Sharpa'],commanded:['#db2777',3,'commanded (after caps)'],
 measured:['#0d9488',3.5,'measured after'],goal0:['#16a34a',3,'goal'],goal1:['#65a30d',3,'lift goal']};
const hidden=new Set(['pred2','pred4','executed']);
function tr(key,pts){const[c,s,n]=STYLE[key];return{type:'scatter3d',mode:'markers',name:n,x:pts.map(p=>p[0]),y:pts.map(p=>p[1]),z:pts.map(p=>p[2]),
 marker:{size:s,color:c,opacity:key==='scene'?0.35:0.9},visible:hidden.has(key)?'legendonly':true,meta:key};}
const layout=()=>({paper_bgcolor:css('--card'),font:{color:css('--fg')},margin:{l:0,r:0,t:0,b:0},legend:{x:0,y:1},uirevision:'keep',
 scene:{aspectmode:'data',xaxis:{title:'x mm'},yaxis:{title:'y mm'},zaxis:{title:'z mm'},bgcolor:css('--card')}});
const slider=document.getElementById('slider');slider.max=D.frames.length-1;
function show(i){const f=D.frames[i];const t=[tr('scene',f.scene)];
 for(const k of ['now','pred1','pred2','pred3','pred4','pred5','executed','fitted','commanded','measured'])if(f.layers[k])t.push(tr(k,f.layers[k]));
 D.goals.forEach((g,j)=>t.push(tr('goal'+j,g)));
 Plotly.react('view',t,layout(),{responsive:true});
 document.getElementById('label').textContent=`replan ${f.replan} · stage ${f.stage} · step ${f.step}${f.reused?' (reused plan)':''} · ${f.distance} mm to goal${f.hold?' · HOLD: '+f.hold:''}`;
 for(const id of ['c_dist','c_shape','c_rot','c_scale'])Plotly.relayout(id,{shapes:[{type:'line',x0:f.replan,x1:f.replan,yref:'paper',y0:0,y1:1,line:{color:css('--muted'),dash:'dot'}}]});}
document.getElementById('view').on?.('plotly_legendclick',()=>true);
const C=D.chart,L=(t,y)=>({title:{text:t,font:{size:13}},paper_bgcolor:css('--card'),plot_bgcolor:css('--card'),font:{color:css('--fg'),size:11},margin:{l:44,r:8,t:28,b:28},
 xaxis:{title:'replan',gridcolor:css('--line')},yaxis:{title:y,gridcolor:css('--line')},legend:{orientation:'h',y:-0.3},showlegend:true});
const line=(y,n,c,d)=>({x:C.replan,y,name:n,mode:'lines+markers',line:{color:c,dash:d||'solid'},marker:{size:4},connectgaps:false});
Plotly.newPlot('c_dist',[line(C.distance,'hand to goal','#2563eb')],L('Distance to goal hand','mm'),{responsive:true});
Plotly.newPlot('c_shape',[line(C.shape_now,'now','#2563eb'),line(C.shape_pred,'predicted','#f97316'),line(C.shape_fit,'fitted','#a855f7','dot'),line(C.shape_cmd,'commanded','#db2777','dash'),line(C.shape_meas,'measured','#0d9488')],
 L('Finger shape error to goal (palm frame)','mm'),{responsive:true});
Plotly.newPlot('c_rot',[line(C.rot_now,'now','#2563eb'),line(C.rot_pred,'predicted','#f97316'),line(C.rot_fit,'fitted','#a855f7','dot'),line(C.rot_cmd,'commanded','#db2777','dash'),line(C.rot_meas,'measured','#0d9488')],
 L('Palm rotation error to goal','deg'),{responsive:true});
Plotly.newPlot('c_scale',[line(C.scale1,'step 1','#fdba74'),line(C.scale3,'step 3','#f97316'),line(C.scale5,'step 5','#9a3412')],
 Object.assign(L('Predicted hand scale (1 = rigid size)','scale'),{yaxis:{title:'scale',gridcolor:css('--line'),range:[0.75,1.1]}}),{responsive:true});
slider.oninput=()=>show(+slider.value);
document.getElementById('prev').onclick=()=>{slider.value=Math.max(0,+slider.value-1);show(+slider.value)};
document.getElementById('next').onclick=()=>{slider.value=Math.min(D.frames.length-1,+slider.value+1);show(+slider.value)};
show(0);
</script></body></html>
"""

if __name__ == "__main__":
    main()
