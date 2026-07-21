#!/usr/bin/env python3
"""Create a single-file WebGL point-splat viewer for TripoSplat PLY files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_triposplat_ply_viewer_html_v5_canvas import build_payload


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title><style>
html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#07090d;color:#eceff4;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
canvas{display:block;width:100vw;height:100vh;cursor:grab;touch-action:none}
canvas:active{cursor:grabbing}
.hud{position:fixed;left:12px;top:12px;z-index:3;max-width:min(620px,calc(100vw - 24px));background:rgba(7,9,13,.74);border:1px solid rgba(255,255,255,.18);border-radius:6px;padding:8px 10px;font-size:12px;line-height:1.45;backdrop-filter:blur(6px);pointer-events:none}
.hud b{font-size:13px}.bar{position:fixed;right:12px;top:12px;z-index:4;display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;pointer-events:auto}
button{height:34px;border:1px solid rgba(255,255,255,.24);background:rgba(255,255,255,.11);color:#fff;border-radius:6px;padding:0 10px;font:12px system-ui;cursor:pointer}
button.on{background:rgba(88,166,255,.34)}
.thumb{position:fixed;left:12px;bottom:12px;z-index:3;width:min(220px,34vw);border:1px solid rgba(255,255,255,.22);border-radius:6px;background:rgba(7,9,13,.72);padding:6px;box-shadow:0 8px 28px rgba(0,0,0,.28);pointer-events:none}
.thumb img{display:block;width:100%;height:auto;border-radius:4px}.thumb.off{display:none}
</style></head><body><canvas id="c"></canvas><div id="thumb" class="thumb off"></div><div class="hud"><b id="title"></b><br><span id="meta"></span></div><div class="bar"><button id="input">Input View</button><button id="spin">Spin</button><button id="reset">Reset</button><button id="quality" class="on">HiDPI</button><button id="depth">Depth</button><button id="thumbbtn">Thumbnail</button></div><script>
const DATA=__DATA__;
const canvas=document.getElementById("c"),titleEl=document.getElementById("title"),metaEl=document.getElementById("meta"),thumb=document.getElementById("thumb");
titleEl.textContent=DATA.title;
metaEl.textContent=DATA.vertices.toLocaleString()+" gaussians / WebGL point-splat / default "+DATA.default_view;
if(DATA.thumbnail_b64){const img=document.createElement("img");img.alt="same-view render";img.src="data:"+DATA.thumbnail_mime+";base64,"+DATA.thumbnail_b64;thumb.appendChild(img);thumb.classList.remove("off")}else{document.getElementById("thumbbtn").style.display="none"}
function decode(b){const bin=atob(b),u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u.buffer}
const POS=new Float32Array(decode(DATA.positions_b64)),COL=new Uint8Array(decode(DATA.colors_b64)),SIZ=new Float32Array(decode(DATA.sizes_b64));
const gl=canvas.getContext("webgl2",{antialias:true,alpha:false,preserveDrawingBuffer:false})||canvas.getContext("webgl",{antialias:true,alpha:false,preserveDrawingBuffer:false});
if(!gl){document.body.innerHTML='<div style="padding:24px;color:#fff;font:16px system-ui">WebGL is not available in this browser.</div>';throw new Error("WebGL unavailable")}
const VS=`attribute vec3 a_pos;attribute vec4 a_col;attribute float a_size;uniform mat4 u_view;uniform mat4 u_proj;uniform float u_dpr;uniform float u_pointScale;uniform float u_pointMax;varying vec4 v_col;void main(){vec4 v=u_view*vec4(a_pos,1.0);gl_Position=u_proj*v;float z=max(-v.z,0.03);gl_PointSize=clamp(a_size*u_pointScale*u_dpr/z,1.0,u_pointMax);v_col=a_col;}`;
const FS=`precision highp float;varying vec4 v_col;void main(){vec2 d=gl_PointCoord-vec2(0.5);float r=length(d);if(r>0.5)discard;float a=exp(-r*r*14.0);if(a<=0.006)discard;gl_FragColor=vec4(v_col.rgb,v_col.a*a*0.36);}`;
function shader(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s}
function program(vs,fs){const p=gl.createProgram();gl.attachShader(p,shader(gl.VERTEX_SHADER,vs));gl.attachShader(p,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));return p}
const prog=program(VS,FS);gl.useProgram(prog);
function buf(data,target=gl.ARRAY_BUFFER){const b=gl.createBuffer();gl.bindBuffer(target,b);gl.bufferData(target,data,gl.STATIC_DRAW);return b}
const bPos=buf(POS),bCol=buf(COL),bSiz=buf(SIZ);
function attr(name,size,type,normalized,stride,offset,buffer){const loc=gl.getAttribLocation(prog,name);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,type,normalized,stride,offset)}
attr("a_pos",3,gl.FLOAT,false,0,0,bPos);attr("a_col",4,gl.UNSIGNED_BYTE,true,0,0,bCol);attr("a_size",1,gl.FLOAT,false,0,0,bSiz);
const uView=gl.getUniformLocation(prog,"u_view"),uProj=gl.getUniformLocation(prog,"u_proj"),uDpr=gl.getUniformLocation(prog,"u_dpr"),uPointScale=gl.getUniformLocation(prog,"u_pointScale"),uPointMax=gl.getUniformLocation(prog,"u_pointMax");
const target=DATA.center,baseRadius=Math.max(DATA.radius,0.01);
const pointRange=gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)||[1,96],pointMax=Math.max(24,Math.min(pointRange[1]||96,192));
const isWebGL2=typeof WebGL2RenderingContext!=="undefined"&&gl instanceof WebGL2RenderingContext;
const canDrawIndexed=DATA.vertices<=65535||isWebGL2||!!gl.getExtension("OES_element_index_uint");
const indexType=DATA.vertices<=65535?gl.UNSIGNED_SHORT:gl.UNSIGNED_INT;
const indexArray=canDrawIndexed?(DATA.vertices<=65535?new Uint16Array(DATA.vertices):new Uint32Array(DATA.vertices)):null;
const indexBuffer=canDrawIndexed?gl.createBuffer():null;
let orderDirty=true,sortedReady=false;
function startDir(){const cp=Math.cos(DATA.default_pitch),sp=Math.sin(DATA.default_pitch),cy=Math.cos(DATA.default_yaw),sy=Math.sin(DATA.default_yaw);return norm([cp*sy,sp,cp*cy])}
let dist=DATA.default_distance,drag=false,lastX=0,lastY=0,spin=false,depth=false,highQuality=true,pointScale=10.5,pending=false,q=[0,0,0,1];
const defaultDir=startDir(),defaultUp=Math.abs(defaultDir[1])>.96?[0,0,1]:[0,1,0];
function schedule(){if(!pending){pending=true;requestAnimationFrame(frame)}}
function inputView(){q=[0,0,0,1];dist=DATA.default_distance;spin=false;orderDirty=true;sortedReady=false;schedule()}
function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}
function norm(a){const l=Math.hypot(a[0],a[1],a[2])||1;return[a[0]/l,a[1]/l,a[2]/l]}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}
function qnorm(a){const l=Math.hypot(a[0],a[1],a[2],a[3])||1;return[a[0]/l,a[1]/l,a[2]/l,a[3]/l]}
function qaxis(axis,ang){const n=norm(axis),s=Math.sin(ang*.5);return[n[0]*s,n[1]*s,n[2]*s,Math.cos(ang*.5)]}
function qmul(a,b){return[a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]]}
function qrot(qq,v){const u=[qq[0],qq[1],qq[2]],t0=cross(u,v),t=[t0[0]*2,t0[1]*2,t0[2]*2],c=cross(u,t);return[v[0]+qq[3]*t[0]+c[0],v[1]+qq[3]*t[1]+c[1],v[2]+qq[3]*t[2]+c[2]]}
function basis(){const z=norm(qrot(q,defaultDir)),up0=norm(qrot(q,defaultUp)),x=norm(cross(up0,z)),y=cross(z,x),eye=[target[0]+dist*z[0],target[1]+dist*z[1],target[2]+dist*z[2]];return{eye,x,y,z}}
function viewMatrix(b){return new Float32Array([b.x[0],b.y[0],b.z[0],0,b.x[1],b.y[1],b.z[1],0,b.x[2],b.y[2],b.z[2],0,-dot(b.x,b.eye),-dot(b.y,b.eye),-dot(b.z,b.eye),1])}
function perspective(fovy,aspect,near,far){const f=1/Math.tan(fovy/2),nf=1/(near-far);return new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)*nf,-1,0,0,2*far*near*nf,0])}
function markOrderDirty(){orderDirty=true}
function applyStableDrag(e){const dx=e.clientX-lastX,dy=e.clientY-lastY;if(!dx&&!dy)return;lastX=e.clientX;lastY=e.clientY;const b=basis(),speed=0.0048;q=qnorm(qmul(qmul(qaxis(b.y,-dx*speed),qaxis(b.x,-dy*speed)),q));markOrderDirty()}
canvas.addEventListener("pointerdown",e=>{drag=true;lastX=e.clientX;lastY=e.clientY;spin=false;canvas.setPointerCapture(e.pointerId);schedule()});
canvas.addEventListener("pointerup",()=>{drag=false;markOrderDirty();schedule()});canvas.addEventListener("pointercancel",()=>{drag=false;markOrderDirty();schedule()});
canvas.addEventListener("pointermove",e=>{if(!drag)return;applyStableDrag(e);schedule()});
canvas.addEventListener("wheel",e=>{e.preventDefault();spin=false;if(e.shiftKey){pointScale=Math.max(2.5,Math.min(48,pointScale*Math.exp(-e.deltaY*.001)))}else{dist*=Math.exp(e.deltaY*.001);dist=Math.max(baseRadius*.35,Math.min(baseRadius*18,dist));markOrderDirty()}schedule()},{passive:false});
canvas.addEventListener("dblclick",inputView);
document.getElementById("input").onclick=inputView;document.getElementById("reset").onclick=inputView;document.getElementById("spin").onclick=()=>{spin=!spin;markOrderDirty();schedule()};
document.getElementById("quality").onclick=e=>{highQuality=!highQuality;e.currentTarget.classList.toggle("on",highQuality);markOrderDirty();schedule()};
document.getElementById("depth").onclick=e=>{depth=!depth;e.currentTarget.classList.toggle("on",depth);markOrderDirty();schedule()};
document.getElementById("thumbbtn").onclick=()=>{thumb.classList.toggle("off")};
function sortForView(b){if(!canDrawIndexed||depth)return false;const n=DATA.vertices,depths=new Float32Array(n),ids=new Array(n);for(let i=0;i<n;i++){const j=i*3;depths[i]=-((POS[j]-b.eye[0])*b.z[0]+(POS[j+1]-b.eye[1])*b.z[1]+(POS[j+2]-b.eye[2])*b.z[2]);ids[i]=i}ids.sort((a,b)=>(depths[b]-depths[a])||(a-b));for(let i=0;i<n;i++)indexArray[i]=ids[i];gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,indexBuffer);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,indexArray,gl.DYNAMIC_DRAW);orderDirty=false;sortedReady=true;return true}
function drawPoints(){if(sortedReady&&!depth){gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,indexBuffer);gl.drawElements(gl.POINTS,DATA.vertices,indexType,0)}else{gl.drawArrays(gl.POINTS,0,DATA.vertices)}}
function frame(){pending=false;if(spin){const b=basis();q=qnorm(qmul(qaxis(b.y,0.006),q))}const rawDpr=devicePixelRatio||1,dpr=highQuality?Math.min(Math.max(rawDpr,drag?1.5:2.0),drag?2.5:4.0):Math.min(rawDpr,drag?1.25:1.5),w=Math.max(1,Math.floor(innerWidth*dpr)),h=Math.max(1,Math.floor(innerHeight*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;gl.viewport(0,0,w,h)}const b=basis();if(highQuality&&!drag&&!spin&&!depth&&orderDirty)sortForView(b);gl.clearColor(0.027,0.035,0.051,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);if(depth){gl.enable(gl.DEPTH_TEST);gl.depthMask(true)}else{gl.disable(gl.DEPTH_TEST);gl.depthMask(false)}gl.uniformMatrix4fv(uView,false,viewMatrix(b));gl.uniformMatrix4fv(uProj,false,perspective(DATA.default_fov_rad,w/h,0.01,Math.max(100,baseRadius*80)));gl.uniform1f(uDpr,dpr);gl.uniform1f(uPointScale,highQuality?pointScale:pointScale*.78);gl.uniform1f(uPointMax,pointMax);drawPoints();if(spin)schedule()}
addEventListener("resize",schedule);
inputView();
</script></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="TripoSplat WebGL viewer")
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
                "renderer": "webgl_pointsplat",
                "single_file": True,
                "has_thumbnail": bool(payload["thumbnail_b64"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
