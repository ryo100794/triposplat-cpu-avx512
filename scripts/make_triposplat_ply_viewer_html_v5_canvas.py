#!/usr/bin/env python3
"""Create a robust single-file Canvas 2D interactive TripoSplat PLY viewer."""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import numpy as np


C0 = 0.28209479177387814


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def read_binary_ply(path: Path) -> dict[str, np.ndarray]:
    data = path.read_bytes()
    marker = b"end_header\n"
    header_end = data.find(marker)
    if header_end < 0:
        raise ValueError(f"PLY header terminator not found: {path}")
    header_end += len(marker)
    header = data[:header_end].decode("ascii", errors="replace").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("Only binary_little_endian PLY is supported")
    count = None
    props: list[str] = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[:2] == ["element", "vertex"]:
            count = int(parts[2])
            in_vertex = True
            continue
        if parts[0] == "element" and parts[1] != "vertex":
            in_vertex = False
        if in_vertex and parts[:2] == ["property", "float"]:
            props.append(parts[2])
    if count is None:
        raise ValueError("PLY vertex count not found")
    dtype = np.dtype([(name, "<f4") for name in props])
    arr = np.frombuffer(data, dtype=dtype, count=count, offset=header_end)
    return {name: np.asarray(arr[name], dtype=np.float32) for name in props}


def b64(a: np.ndarray) -> str:
    return base64.b64encode(a.tobytes()).decode("ascii")


def image_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def default_angles(view: str) -> tuple[float, float]:
    if view == "front_z":
        return 0.0, 0.0
    if view == "back_z":
        return math.pi, 0.0
    if view == "front_x":
        return math.pi * 0.5, 0.0
    if view == "back_x":
        return -math.pi * 0.5, 0.0
    if view == "front_y":
        return 0.0, math.pi * 0.49
    if view == "back_y":
        return 0.0, -math.pi * 0.49
    raise ValueError(f"Unknown default view: {view}")


def build_payload(
    ply: Path,
    title: str,
    point_scale: float,
    default_view: str,
    fov_deg: float,
    distance_scale: float,
    thumbnail_image: Path | None,
) -> dict:
    p = read_binary_ply(ply)
    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2"}
    missing = sorted(required.difference(p))
    if missing:
        raise ValueError(f"PLY is missing fields: {missing}")
    pos = np.stack([p["x"], p["y"], p["z"]], axis=1).astype("<f4")
    rgb = np.clip((np.stack([p["f_dc_0"], p["f_dc_1"], p["f_dc_2"]], axis=1) * C0 + 0.5) * 255, 0, 255)
    alpha = np.clip(sigmoid(p["opacity"]) * 255, 32, 255)[:, None]
    rgba = np.concatenate([rgb.astype(np.uint8), alpha.astype(np.uint8)], axis=1)
    scales = np.exp(np.stack([p["scale_0"], p["scale_1"], p["scale_2"]], axis=1)).astype(np.float32)
    raw_size = np.mean(scales, axis=1)
    ref = float(np.percentile(raw_size, 90))
    if ref <= 0:
        ref = 1.0
    sizes = np.clip(raw_size / ref * point_scale, 1.2, 28.0).astype("<f4")
    center = np.median(pos, axis=0).astype(np.float32)
    radius = float(np.percentile(np.linalg.norm(pos - center[None, :], axis=1), 95))
    if radius <= 1e-6:
        radius = float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0)) * 0.5 + 1e-3)
    yaw, pitch = default_angles(default_view)
    payload = {
        "title": title,
        "vertices": int(pos.shape[0]),
        "positions_b64": b64(pos),
        "colors_b64": b64(rgba),
        "sizes_b64": b64(sizes),
        "center": center.astype(float).tolist(),
        "radius": radius,
        "source_ply": ply.as_posix(),
        "default_view": default_view,
        "default_yaw": yaw,
        "default_pitch": pitch,
        "default_distance": float(radius * distance_scale),
        "default_fov_rad": float(math.radians(fov_deg)),
        "thumbnail_mime": "",
        "thumbnail_b64": "",
    }
    if thumbnail_image is not None:
        payload["thumbnail_mime"] = image_mime(thumbnail_image)
        payload["thumbnail_b64"] = base64.b64encode(thumbnail_image.read_bytes()).decode("ascii")
    return payload


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title><style>
html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#07090d;color:#eceff4;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
canvas{display:block;width:100vw;height:100vh;cursor:grab;touch-action:none}
canvas:active{cursor:grabbing}
.hud{position:fixed;left:12px;top:12px;z-index:3;max-width:min(560px,calc(100vw - 24px));background:rgba(7,9,13,.72);border:1px solid rgba(255,255,255,.18);border-radius:6px;padding:8px 10px;font-size:12px;line-height:1.45;backdrop-filter:blur(6px);pointer-events:none}
.hud b{font-size:13px}.bar{position:fixed;right:12px;top:12px;z-index:4;display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;pointer-events:auto}
button{height:34px;border:1px solid rgba(255,255,255,.24);background:rgba(255,255,255,.11);color:#fff;border-radius:6px;padding:0 10px;font:12px system-ui;cursor:pointer}
.thumb{position:fixed;left:12px;bottom:12px;z-index:3;width:min(220px,34vw);border:1px solid rgba(255,255,255,.22);border-radius:6px;background:rgba(7,9,13,.72);padding:6px;box-shadow:0 8px 28px rgba(0,0,0,.28);pointer-events:none}
.thumb img{display:block;width:100%;height:auto;border-radius:4px}.thumb.off{display:none}
</style></head><body><canvas id="c"></canvas><div id="thumb" class="thumb off"></div><div class="hud"><b id="title"></b><br><span id="meta"></span></div><div class="bar"><button id="input">Input View</button><button id="spin">Spin</button><button id="reset">Reset</button><button id="thumbbtn">Thumbnail</button></div><script>
const DATA=__DATA__;
const canvas=document.getElementById("c"),ctx=canvas.getContext("2d"),titleEl=document.getElementById("title"),metaEl=document.getElementById("meta"),thumb=document.getElementById("thumb");
titleEl.textContent=DATA.title;
metaEl.textContent=DATA.vertices.toLocaleString()+" gaussians / Canvas 2D / default "+DATA.default_view;
if(DATA.thumbnail_b64){const img=document.createElement("img");img.alt="same-view render";img.src="data:"+DATA.thumbnail_mime+";base64,"+DATA.thumbnail_b64;thumb.appendChild(img);thumb.classList.remove("off")}else{document.getElementById("thumbbtn").style.display="none"}
function decode(b){const bin=atob(b),u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u.buffer}
const P=new Float32Array(decode(DATA.positions_b64)),C=new Uint8Array(decode(DATA.colors_b64)),S=new Float32Array(decode(DATA.sizes_b64));
const target=DATA.center,baseRadius=Math.max(DATA.radius,0.01);
let yaw=DATA.default_yaw,pitch=DATA.default_pitch,dist=DATA.default_distance,drag=false,lastX=0,lastY=0,spin=false;
function inputView(){yaw=DATA.default_yaw;pitch=DATA.default_pitch;dist=DATA.default_distance;spin=false}
function reset(){inputView()}
function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}
function norm(a){const l=Math.hypot(a[0],a[1],a[2])||1;return[a[0]/l,a[1]/l,a[2]/l]}
function cameraBasis(){const cp=Math.cos(pitch),sp=Math.sin(pitch),cy=Math.cos(yaw),sy=Math.sin(yaw),eye=[target[0]+dist*cp*sy,target[1]+dist*sp,target[2]+dist*cp*cy];const z=norm([eye[0]-target[0],eye[1]-target[1],eye[2]-target[2]]),up=Math.abs(z[1])>.96?[0,0,1]:[0,1,0],x=norm(cross(up,z)),y=cross(z,x);return{eye,x,y,z}}
canvas.addEventListener("pointerdown",e=>{drag=true;lastX=e.clientX;lastY=e.clientY;spin=false;canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener("pointerup",()=>{drag=false});canvas.addEventListener("pointercancel",()=>{drag=false});
canvas.addEventListener("pointermove",e=>{if(!drag)return;yaw+=(e.clientX-lastX)*0.006;pitch=Math.max(-1.50,Math.min(1.50,pitch+(e.clientY-lastY)*0.006));lastX=e.clientX;lastY=e.clientY});
canvas.addEventListener("wheel",e=>{e.preventDefault();spin=false;dist*=Math.exp(e.deltaY*0.001);dist=Math.max(baseRadius*0.35,Math.min(baseRadius*18,dist))},{passive:false});
canvas.addEventListener("dblclick",inputView);document.getElementById("input").onclick=inputView;document.getElementById("reset").onclick=reset;document.getElementById("spin").onclick=()=>{spin=!spin};document.getElementById("thumbbtn").onclick=()=>{thumb.classList.toggle("off")};
function draw(){const dpr=Math.min(devicePixelRatio||1,2),w=Math.max(1,Math.floor(innerWidth*dpr)),h=Math.max(1,Math.floor(innerHeight*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}if(spin)yaw+=0.006;ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle="#07090d";ctx.fillRect(0,0,w,h);const cam=cameraBasis(),f=(Math.min(w,h)*0.5)/Math.tan(DATA.default_fov_rad*0.5),items=[];for(let i=0,n=DATA.vertices;i<n;i++){const j=i*3,dx=P[j]-cam.eye[0],dy=P[j+1]-cam.eye[1],dz=P[j+2]-cam.eye[2],cx=dx*cam.x[0]+dy*cam.x[1]+dz*cam.x[2],cy=dx*cam.y[0]+dy*cam.y[1]+dz*cam.y[2],cz=dx*cam.z[0]+dy*cam.z[1]+dz*cam.z[2],depth=-cz;if(depth<=0.01)continue;const sx=w*0.5+f*cx/depth,sy=h*0.5-f*cy/depth;if(sx<-80||sx>w+80||sy<-80||sy>h+80)continue;items.push([depth,sx,sy,i])}items.sort((a,b)=>b[0]-a[0]);for(const it of items){const i=it[3],ci=i*4,depth=it[0],r=Math.max(1.0,Math.min(18.0,S[i]*2.2/depth))*dpr,alpha=Math.max(.28,C[ci+3]/255);ctx.globalAlpha=alpha;ctx.fillStyle=`rgb(${C[ci]},${C[ci+1]},${C[ci+2]})`;ctx.fillRect(it[1]-r*.5,it[2]-r*.5,r,r)}ctx.globalAlpha=1;requestAnimationFrame(draw)}
inputView();draw();
</script></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="TripoSplat PLY viewer")
    parser.add_argument("--point-scale", type=float, default=6.5)
    parser.add_argument("--default-view", default="back_z", choices=["front_z", "back_z", "front_x", "back_x", "front_y", "back_y"])
    parser.add_argument("--fov-deg", type=float, default=38.0)
    parser.add_argument("--distance-scale", type=float, default=2.8)
    parser.add_argument("--thumbnail-image", type=Path)
    args = parser.parse_args()
    payload = build_payload(
        args.ply,
        args.title,
        args.point_scale,
        args.default_view,
        args.fov_deg,
        args.distance_scale,
        args.thumbnail_image,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        HTML.replace("__TITLE__", args.title).replace("__DATA__", json.dumps(payload, separators=(",", ":"))),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "vertices": payload["vertices"],
                "source_ply": payload["source_ply"],
                "default_view": payload["default_view"],
                "fov_deg": args.fov_deg,
                "distance_scale": args.distance_scale,
                "has_thumbnail": bool(payload["thumbnail_b64"]),
                "renderer": "canvas2d",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
