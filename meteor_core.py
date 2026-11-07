"""Core code for the Meteor Scatter 3D Explorer notebook.

Geometry and Doppler functions (from meteor_doppler_wdpn.py), the HTML/JavaScript page template
(from meteor_3d_interactive.py), and the plot, table and page-building helpers. The notebook form in
meteor_ui.py calls configure() and then these functions. Most users never need to edit this file.
"""
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import get_plotlyjs
from IPython.display import IFrame

PAGE_FILE = "meteor_3d_interactive_form.html"   # written next to the notebook by build_interactive()
# Map drawn on the ground of the 3D page: a web-Mercator tile server ({z}/{y}/{x}), its credit line, and a
# folder where downloaded tiles are kept so later runs work offline. Set MAP_TILES = None for plain green ground.
MAP_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
MAP_CREDIT = "Map tiles © Esri"
TILE_CACHE = Path(".tile_cache")

# ---- Constants and current setup ------------------------------------------------------------
A, E2 = 6378137.0, 0.00669437999014   # WGS-84 semi-major axis (m) and first eccentricity squared
C = 299792458.0                       # speed of light (m/s)
LIM = 100.0                           # Doppler limit (Hz) defining the "near-zero Doppler" band

def configure(tx, rx, freq_mhz, speed_kms, alt_km=100.0, lim_hz=100.0):
    """Set the site and signal values the functions below read as module globals.

    tx, rx: (lat deg, lon deg, height m).  speed_kms and alt_km are used by the static plots and tables."""
    global TX, RX, F, LAM, V_KMS, V, ALT_KM, LIM
    TX, RX = tuple(tx), tuple(rx)
    F = freq_mhz * 1e6              # carrier frequency (Hz)
    LAM = C / F                     # wavelength (m)
    V_KMS = speed_kms               # meteor speed for the static plots (km/s)
    V = V_KMS * 1e3                 # speed (m/s)
    ALT_KM = alt_km                 # reflection-point altitude for the static plots and tables (km)
    LIM = lim_hz                    # near-zero-Doppler limit (Hz)


# Defaults: WDPN (Philadelphia, ch 2 pilot) to a receiver near Washington DC
configure((40.04169, -75.23614, 378.9), (38.821708, -77.103720, 100.0), 54.309441, 20.0)


# ---- Geometry functions -----------------------------------------------------------------------
def ecef(lat, lon, h):
    """Convert geodetic lat/lon (deg) and height h (m) to Earth-centered Earth-fixed xyz (m)."""
    la, lo = np.radians(lat), np.radians(lon)
    n = A / np.sqrt(1 - E2 * np.sin(la) ** 2)   # prime-vertical radius of curvature
    return np.array([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                     (n * (1 - E2) + h) * np.sin(la)])


def enu(lat, lon):
    """Local East, North, Up unit vectors (expressed in ECEF coordinates) at lat/lon (deg)."""
    la, lo = np.radians(lat), np.radians(lon)
    e = np.array([-np.sin(lo), np.cos(lo), 0.0])                                  # east
    n = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])  # north
    u = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return e, n, u


def geodetic(p, alt_m):
    """(lat, lon, alt) of the point straight above direction p (geocentric direction -> geodetic lat)."""
    lon = np.degrees(np.arctan2(p[1], p[0]))
    lat = np.degrees(np.arctan2(p[2], (1 - E2) * np.hypot(p[0], p[1])))
    return lat, lon, alt_m


def point_on_path(frac, alt_m):
    """Point above the great-circle-ish chord TX->RX at fraction frac, altitude alt (m)."""
    # linear interpolation between the TX and RX ground points (frac=0 -> TX, 1 -> RX)
    g = (1 - frac) * ecef(*TX[:2], 0) + frac * ecef(*RX[:2], 0)
    g /= np.linalg.norm(g)   # project the chord point back out to a direction from Earth's center
    # lat/lon of that direction
    return geodetic(g, alt_m)


def bisector(P):
    """For reflection point P (lat, lon, alt), return (b, uT, uR): b = uT + uR, the (unnormalised)
    bisector of the unit vectors pointing from TX and RX to P. Doppler depends only on vhat . b."""
    p = ecef(*P)                                          # meteor position in ECEF
    uT = (p - ecef(*TX)); uT /= np.linalg.norm(uT)        # unit vector TX -> P
    uR = (p - ecef(*RX)); uR /= np.linalg.norm(uR)        # unit vector RX -> P
    return uT + uR, uT, uR


def doppler(P, az_deg, desc_deg):
    """Doppler (Hz) for velocity heading az (deg from N, direction of travel) descending desc deg below horizontal."""
    e, n, u = enu(P[0], P[1])
    az, d = np.radians(az_deg), np.radians(desc_deg)
    # unit velocity vector in ECEF: horizontal part split into east/north by heading, minus the descent along "up"
    # (az_deg and desc_deg may be arrays; [..., None] lets them broadcast against the 3-vectors)
    vhat = (np.cos(d) * np.sin(az))[..., None] * e + (np.cos(d) * np.cos(az))[..., None] * n \
        - np.sin(d)[..., None] * u
    b, _, _ = bisector(P)
    return -(V / LAM) * (vhat @ b)   # f_D = -(v/lambda) vhat . (uT + uR); positive = path shortening


def zero_doppler_desc(P, az_deg):
    """Descent angle (deg) giving f_D=0 at this heading, or nan if none in 0..90."""
    e, n, u = enu(P[0], P[1])
    b, _, _ = bisector(P)
    be, bn, bu = b @ e, b @ n, b @ u               # east/north/up components of the bisector
    az = np.radians(az_deg)
    h = be * np.sin(az) + bn * np.cos(az)          # horizontal component of b along heading
    # cos(d)*h - sin(d)*bu = 0 -> tan d = h/bu
    d = np.degrees(np.arctan2(h, bu))
    # fold the angle into -90..90 (tan has period 180 deg), then keep only physical descents 0..90
    d = np.where(d < -90, d + 180, np.where(d > 90, d - 180, d))
    return np.where((d >= 0) & (d <= 90), d, np.nan)


def band_halfwidth(P, az_deg, f0=0.0):
    """Range of descent angles giving |f_D|<=LIM at a heading: returns (lo, hi) arrays (deg)."""
    dd = np.linspace(0, 90, 9001)      # descent angles sampled at 0.01 deg steps
    out = []                           # one (lo, hi) pair per heading
    for a in np.atleast_1d(az_deg):
        f = doppler(P, np.full_like(dd, a), dd)   # Doppler along the descent sweep at this heading
        ok = np.abs(f) <= LIM                     # mask of descents inside the +-LIM band
        out.append((dd[ok].min(), dd[ok].max()) if ok.any() else (np.nan, np.nan))
    return np.array(out)


def P_offset(x_km, y_km, alt_m):
    """Reflection point x_km along the baseline (+ toward TX) and y_km to the side (+ = toward 320 deg, NW) of the midpoint."""
    g0 = ecef(*point_on_path(0.5, 0)[:2], 0)        # ground point at the baseline midpoint
    e, n, u = enu(*point_on_path(0.5, 0)[:2])       # local up (u) at the midpoint
    t = ecef(*TX[:2], 0) - ecef(*RX[:2], 0); t /= np.linalg.norm(t)   # unit vector along baseline, RX -> TX
    side = np.cross(u, t); side /= np.linalg.norm(side)               # horizontal unit vector perpendicular to baseline
    p = g0 + x_km * 1e3 * t + y_km * 1e3 * side     # displaced ground point (ECEF)
    return geodetic(p / np.linalg.norm(p), alt_m)


def heading_windows(P, desc_deg):
    """Headings (deg) that give f_D=0 and the +-LIM edges, for a given descent angle. Returns list of (az0, az_lo, az_hi)."""
    e, n, u = enu(P[0], P[1])
    b, _, _ = bisector(P)
    be, bn, bu = b @ e, b @ n, b @ u
    bh, azb = np.hypot(be, bn), np.degrees(np.arctan2(be, bn))   # horizontal magnitude and azimuth of b
    d = np.radians(desc_deg)
    # f_D = 0 requires cos(heading - azb) = c0; the +-LIM edges shift that cosine by +-dc
    c0 = bu * np.tan(d) / bh
    dc = LIM * LAM / (V * np.cos(d)) / bh
    res = []
    for sgn in (+1, -1):               # the two mirror-image headings on either side of azb
        out = []
        for c in (c0, c0 - dc, c0 + dc):   # centre (0 Hz), then the two band edges
            # nan if no heading satisfies it (|cos| > 1)
            out.append((azb + sgn * np.degrees(np.arccos(c))) % 360 if abs(c) <= 1 else np.nan)
        res.append(tuple(out))
    return res


# ---- Interactive page template ------------------------------------------------------------
# __PLOTLY__ and __K__ are filled in by build_interactive(); scale_page() rewrites the site-dependent
# numbers and labels using exact text replacements, so edit those together.
PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Meteor Scatter 3D</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--card:#f3f4f6;--ok:#1a7f37;--bad:#b42318}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e6e6;--mut:#9aa0a6;--card:#1f2329;--ok:#4cc26a;--bad:#ff7b72}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
#wrap{display:flex;flex-wrap:wrap;gap:12px;padding:10px}
#ctl{flex:0 0 300px;background:var(--card);border-radius:8px;padding:12px}
#plots{flex:1 1 600px;min-width:320px}
label{display:flex;justify-content:space-between;margin-top:8px}label b{font-weight:600}
input[type=range]{width:100%}
button{margin:4px 4px 0 0;padding:4px 8px;border:1px solid var(--mut);background:var(--bg);color:var(--fg);border-radius:5px;cursor:pointer}
#fd{font-size:26px;font-weight:700;margin:6px 0 2px}.ok{color:var(--ok)}.bad{color:var(--bad)}
small{color:var(--mut)}h1{font-size:15px;margin:0 0 6px}
</style></head><body><div id="wrap">
<div id="ctl"><h1>WDPN 54.31 MHz → RX forward scatter</h1>
<div id="fd">–</div><div id="fdinfo"><small></small></div>
<div>Presets:<br><button data-p="q">¼ from RX</button><button data-p="m">Midpoint</button><button data-p="t">¼ from TX</button></div>
<label>Reflection point along path <b id="vx"></b></label><input id="x" type="range" min="-105" max="105" step="1" value="-52">
<label>Offset from path (+ = NW) <b id="vy"></b></label><input id="y" type="range" min="-250" max="250" step="5" value="0">
<label>Altitude <b id="vh"></b></label><input id="h" type="range" min="70" max="130" step="1" value="100">
<label>Heading (direction of travel, from N) <b id="vaz"></b></label><input id="az" type="range" min="0" max="359" step="1" value="240">
<label>Descent below horizontal <b id="vd"></b></label><input id="d" type="range" min="0" max="90" step="0.1" value="14.5">
<label>Speed <b id="vv"></b></label><input id="v" type="range" min="11" max="72" step="1" value="20">
<div><button id="snap">Set descent for 0 Hz at this heading</button></div>
<div><label style="justify-content:flex-start;gap:6px"><input id="ex" type="checkbox"> vertical exaggeration ×2</label></div>
<div style="margin-top:8px"><small>Blue shell = surface of equal TX→P→RX path length through the reflection point (the 0 Hz "tangent" surface for that point). Green path: |f<sub>D</sub>| ≤ 100 Hz at P; red: outside. Convention: + Hz = path shortening. Straight-line paths, no refraction. Ground curvature is real. The view is centered on RX. <span id="credit"></span></small></div>
</div>
<div id="plots"><div id="g3" style="height:560px"></div><div id="g2" style="height:240px"></div></div></div>
<script>__PLOTLY__</script>
<script>
// K = constants injected from Python (station positions, Earth center, wavelength); D2R converts degrees to radians
const K=__K__; const D2R=Math.PI/180;
// small 3-vector helpers: subtract, add, scale, dot, cross, norm, unit vector
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]],add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]],sc=(a,s)=>[a[0]*s,a[1]*s,a[2]*s];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2],cr=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const nrm=a=>Math.sqrt(dot(a,a)),un=a=>sc(a,1/nrm(a));
// TX, RX, baseline midpoint (on the ground) and polar axis, all in the page frame (km); the RX ground point is the origin
const T=K.T,R=K.R,M=K.M,Z=K.Z;
const upM=un(sub(M,K.C));   // local up at the midpoint
// tb: horizontal unit vector along the baseline (RX->TX); sd: horizontal unit vector perpendicular to it (the "offset" direction)
const tb=un(sub(sub(T,R),sc(upM,dot(sub(T,R),upM)))), sd=cr(upM,tb);
// local frame at point P: up = away from Earth's center, e = east, n = north (all unit vectors)
function frame(P){const up=un(sub(P,K.C));const e=un(cr(Z,up));const n=cr(up,e);return{up,e,n}}
// reflection point x km along the baseline and y km to the side of the midpoint, h km above the curved ground
function place(x,y,h){const P0=add(M,add(sc(tb,x),sc(sd,y)));return add(K.C,sc(un(sub(P0,K.C)),K.RE+h))}
// unit velocity vector at P for heading az (deg from N) and descent d (deg below horizontal)
function vel(P,az,d){const f=frame(P),a=az*D2R,dd=d*D2R;
  return add(add(sc(f.e,Math.cos(dd)*Math.sin(a)),sc(f.n,Math.cos(dd)*Math.cos(a))),sc(f.up,-Math.sin(dd)))}
// bisector b = uT + uR (unit vectors from TX and RX to P)
function bis(P){return add(un(sub(P,T)),un(sub(P,R)))}
// Doppler (Hz) for unit velocity v at P and speed vkms (km/s); + = path shortening
function fD(P,v,vkms){return -(vkms*1000/K.LAM)*dot(v,bis(P))}
// Ellipsoid of constant TX->P->RX path length S with foci at TX and RX; returns surface grids X,Y,Zz plus S, a, b
// (a, b = semi-major / semi-minor axes). Points below ground are set to null so they are not drawn.
function ellipsoid(P){const S=nrm(sub(P,T))+nrm(sub(P,R)),c=sc(add(T,R),.5),ax=un(sub(T,R)),f=nrm(sub(T,R))/2,a=S/2,b=Math.sqrt(a*a-f*f);
  // c = center, ax = major axis, f = focal half-distance; e1, e2 span the plane perpendicular to the axis;
  // nt / np_ = number of latitude / longitude grid steps
  const e1=un(cr(ax,[0,0,1])),e2=cr(ax,e1),nt=50,np_=72,X=[],Y=[],Zz=[];
  for(let i=0;i<=nt;i++){const th=Math.PI*i/nt,rx=X[i]=[],ry=Y[i]=[],rz=Zz[i]=[];   // th: angle from the major axis
    for(let j=0;j<=np_;j++){const ph=2*Math.PI*j/np_;   // ph: angle around the major axis
      const p=add(add(c,sc(ax,a*Math.cos(th))),sc(add(sc(e1,Math.cos(ph)),sc(e2,Math.sin(ph))),b*Math.sin(th)));
      const under=nrm(sub(p,K.C))<K.RE;rx.push(under?null:p[0]);ry.push(under?null:p[1]);rz.push(under?null:p[2]);}}
  return{X,Y,Zz,S,a,b}}
// ground: (GN+1)x(GN+1) grid over +-K.G km East/North of RX, lifted onto the sphere of radius RE about Earth's center,
// drawn as a mesh of two triangles per cell. K.MAP holds one map colour per cell (6 hex digits each), or is null for plain green.
const GN=K.GN,gx=[],gy=[],gz=[],gi=[],gj=[],gk=[],gc=[];
for(let i=0;i<=GN;i++)for(let j=0;j<=GN;j++){const x=-K.G+2*K.G*j/GN,y=-K.G+2*K.G*i/GN,dx=x-K.C[0],dy=y-K.C[1];
  gx.push(x);gy.push(y);gz.push(K.C[2]+Math.sqrt(K.RE*K.RE-dx*dx-dy*dy))}
for(let i=0;i<GN;i++)for(let j=0;j<GN;j++){const a=i*(GN+1)+j,b=a+1,c=a+GN+1,d=c+1;gi.push(a,b);gj.push(b,d);gk.push(c,c);
  if(K.MAP){const h='#'+K.MAP.substr(6*(i*GN+j),6);gc.push(h,h)}}
const gnd=K.MAP?{facecolor:gc,opacity:1}:{color:'#556b2f',opacity:.25};
document.getElementById('credit').textContent=K.MAP?K.CREDIT:'';
const col=s=>[[0,s],[1,s]];   // single-colour colorscale
// 3-D traces. Indices matter: update() restyles 1 (ellipsoid), 5 (path), 6 (arrowhead), 7 (RX-P-TX lines), 8 (point P)
const tr=[
 {type:'mesh3d',x:gx,y:gy,z:gz,i:gi,j:gj,k:gk,...gnd,flatshading:true,lighting:{ambient:1,diffuse:0,specular:0,fresnel:0},hoverinfo:'skip',name:'ground'},
 {type:'surface',x:[[0]],y:[[0]],z:[[0]],showscale:false,opacity:.33,colorscale:col('#4c78a8'),hoverinfo:'skip',name:'equal-path ellipsoid'},
 {type:'scatter3d',mode:'markers+text',x:[T[0]],y:[T[1]],z:[T[2]],text:['WDPN TX'],textposition:'top center',marker:{size:6,color:'red'},name:'TX'},
 {type:'scatter3d',mode:'markers+text',x:[R[0]],y:[R[1]],z:[R[2]],text:['RX'],textposition:'top center',marker:{size:6,color:'blue'},name:'RX'},
 {type:'scatter3d',mode:'lines',x:[T[0],R[0]],y:[T[1],R[1]],z:[T[2],R[2]],line:{color:'gray',width:3,dash:'dash'},name:'baseline'},
 {type:'scatter3d',mode:'lines',x:[0,0],y:[0,0],z:[0,0],line:{color:'green',width:7},name:'meteor path'},
 {type:'cone',x:[0],y:[0],z:[0],u:[1],v:[0],w:[0],sizemode:'absolute',sizeref:14,anchor:'tip',showscale:false,colorscale:col('green'),hoverinfo:'skip'},
 {type:'scatter3d',mode:'lines',x:[0,0],y:[0,0],z:[0,0],line:{color:'rgba(120,120,120,.8)',width:2},name:'RX→P→TX'},
 {type:'scatter3d',mode:'markers',x:[0],y:[0],z:[0],marker:{size:6,color:'black'},name:'reflection point'}];
const dark=matchMedia('(prefers-color-scheme:dark)').matches;   // match plot text to the page theme
// layout for the 3-D plot (lay) and, below, for the Doppler-vs-distance plot (lay2)
const lay={margin:{l:0,r:0,t:0,b:0},showlegend:false,paper_bgcolor:'rgba(0,0,0,0)',font:{color:dark?'#ddd':'#222'},
 scene:{xaxis:{title:{text:'East km'},range:[-K.G,K.G]},yaxis:{title:{text:'North km'},range:[-K.G,K.G]},zaxis:{title:{text:'Up km'},range:K.ZR},
  aspectmode:'manual',aspectratio:K.AR,camera:{eye:{x:1.2,y:-1.6,z:.8}}}};   // fixed ranges keep RX (the origin) in the center
Plotly.newPlot('g3',tr,lay,{responsive:true});
const lay2={margin:{l:55,r:10,t:22,b:36},title:{text:'Doppler along the trail (the meteor moves →)',font:{size:12}},xaxis:{title:'distance along trail from reflection point P, km'},yaxis:{title:'f_D, Hz'},
 paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:dark?'#ddd':'#222'},showlegend:false,
 shapes:[{type:'rect',xref:'paper',x0:0,x1:1,yref:'y',y0:-100,y1:100,fillcolor:'rgba(26,127,55,.15)',line:{width:0}},{type:'line',x0:0,x1:0,yref:'paper',y0:0,y1:1,line:{dash:'dot',color:'gray'}}]};
Plotly.newPlot('g2',[{x:[0],y:[0],mode:'lines',line:{color:'#d62728'}}],lay2,{responsive:true});
const $=id=>document.getElementById(id);   // shorthand for getElementById
// slider ids and the unit shown next to each value label
const ids=['x','y','h','az','d','v'],unit={x:' km',y:' km',h:' km',az:'°',d:'°',v:' km/s'};
// recompute everything from the sliders and redraw both plots and the readout
function update(){
  // x: along path, y: offset, h: altitude, az: heading, d: descent, vk: speed
  const x=+$('x').value,y=+$('y').value,h=+$('h').value,az=+$('az').value,d=+$('d').value,vk=+$('v').value;
  ids.forEach(i=>$('v'+i).textContent=$(i).value+unit[i]);
  // P: reflection point, v: unit velocity, f: Doppler (Hz), E: ellipsoid through P
  const P=place(x,y,h),v=vel(P,az,d),f=fD(P,v,vk),E=ellipsoid(P);
  const ok=Math.abs(f)<=100,L=60;   // ok: within +-100 Hz; L: half-length of drawn trail (km)
  const a0=sub(P,sc(v,L)),a1=add(P,sc(v,L));   // trail endpoints, a1 = leading end
  Plotly.restyle('g3',{x:[E.X],y:[E.Y],z:[E.Zz]},[1]);
  const c=ok?'#1a7f37':'#d62728';
  Plotly.restyle('g3',{x:[[a0[0],a1[0]]],y:[[a0[1],a1[1]]],z:[[a0[2],a1[2]]],'line.color':c},[5]);
  Plotly.restyle('g3',{x:[[a1[0]]],y:[[a1[1]]],z:[[a1[2]]],u:[[v[0]]],v:[[v[1]]],w:[[v[2]]],colorscale:[col(c)]},[6]);
  Plotly.restyle('g3',{x:[[R[0],P[0],T[0]]],y:[[R[1],P[1],T[1]]],z:[[R[2],P[2],T[2]]]},[7]);
  Plotly.restyle('g3',{x:[[P[0]]],y:[[P[1]]],z:[[P[2]]]},[8]);
  $('fd').textContent=(f>=0?'+':'')+f.toFixed(0)+' Hz';$('fd').className=ok?'ok':'bad';
  const b=bis(P),fr=frame(P);   // bisector and local frame, for the info text
  $('fdinfo').firstChild.innerHTML=`at reflection point; max possible ±${(vk*1000/K.LAM*nrm(b)).toFixed(0)} Hz at ${vk} km/s.<br>Path length TX→P→RX = ${E.S.toFixed(1)} km; ranges ${nrm(sub(P,R)).toFixed(0)} km (RX), ${nrm(sub(P,T)).toFixed(0)} km (TX).<br>Bisector elevation ${(Math.asin(dot(b,fr.up)/nrm(b))/D2R).toFixed(0)}°.`;
  // Doppler sampled every km along the trail (s = signed distance from P), skipping points below ground
  const xs=[],ys=[];for(let s=-L;s<=L;s+=1){const Q=add(P,sc(v,s));if(nrm(sub(Q,K.C))<K.RE){continue}xs.push(s);ys.push(fD(Q,v,vk))}
  Plotly.react('g2',[{x:xs,y:ys,mode:'lines',line:{color:'#d62728',width:2}}],lay2);
}
ids.forEach(i=>$(i).addEventListener('input',update));   // redraw whenever any slider moves
// "Set descent for 0 Hz": same solution as zero_doppler_desc() in meteor_doppler_wdpn.py
// (hh = horizontal part of the bisector along the heading, bu = its vertical part; tan d = hh/bu)
$('snap').onclick=()=>{const x=+$('x').value,y=+$('y').value,h=+$('h').value,az=+$('az').value;
  const P=place(x,y,h),b=bis(P),f=frame(P),hh=dot(b,f.e)*Math.sin(az*D2R)+dot(b,f.n)*Math.cos(az*D2R),bu=dot(b,f.up);
  let d=Math.atan2(hh,bu)/D2R;if(d<-90)d+=180;if(d>90)d-=180;
  if(d>=0&&d<=90){$('d').value=d.toFixed(1);update()}else{$('fdinfo').firstChild.innerHTML='<b>No 0 Hz descent at this heading</b> (it would have to climb: '+d.toFixed(1)+'°). Try another heading.'}};
// preset buttons: put the reflection point 1/4 from RX, at the midpoint, or 1/4 from TX, on the baseline
document.querySelectorAll('button[data-p]').forEach(b=>b.onclick=()=>{$('x').value={q:-52,m:0,t:52}[b.dataset.p];$('y').value=0;update()});
// vertical exaggeration checkbox: double the true-scale height of the box (K.AR)
$('ex').onchange=e=>Plotly.relayout('g3',{'scene.aspectratio':e.target.checked?{...K.AR,z:2*K.AR.z}:K.AR});
update();
</script></body></html>"""


# ---- Plots, tables and page builder ----------------------------------------------------------
def baseline_between(tx, rx):
    """Ground distance (km) between two (lat, lon, ...) sites."""
    return np.linalg.norm(ecef(*tx[:2], 0) - ecef(*rx[:2], 0)) / 1e3


def baseline_km():
    """Ground distance between the configured TX and RX (km)."""
    return baseline_between(TX, RX)


def bearing_deg(a, b):
    """Initial great-circle bearing (deg from north) from site a to site b, each (lat, lon, ...)."""
    la1, la2 = np.radians(a[0]), np.radians(b[0])
    dlo = np.radians(b[1] - a[1])
    y = np.sin(dlo) * np.cos(la2)
    x = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlo)
    return float(np.degrees(np.arctan2(y, x)) % 360)


def compass(bearing_deg):
    """8-point compass name (N, NE, ...) for a bearing in degrees from north."""
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return names[int(((bearing_deg % 360) + 22.5) // 45) % 8]


def print_summary(tx_name="TX", rx_name="RX"):
    """Plain-language summary of the setup and of how restrictive the near-zero-Doppler condition is."""
    base = baseline_km()
    brg = bearing_deg(TX, RX)
    P0 = point_on_path(0.5, ALT_KM * 1e3)        # reference reflection point: baseline midpoint
    b, uT, uR = bisector(P0)
    k = LIM * LAM / (V * np.linalg.norm(b))      # limit on |vhat . bhat| for |f_D| <= LIM
    print(f"{tx_name} (TX) to {rx_name} (RX): ground baseline {base:.1f} km; the receiver is at "
          f"bearing {brg:.0f} deg ({compass(brg)}) from the transmitter.")
    print(f"Carrier {F/1e6:.6g} MHz (wavelength {LAM:.2f} m). The static plots and tables assume "
          f"a meteor speed of {V_KMS:g} km/s and a reflection point {ALT_KM:g} km up.")
    print()
    print(f"At the midpoint of the baseline, no trail can produce more than +-{V/LAM*np.linalg.norm(b):.0f} Hz of Doppler at this speed.")
    print(f"A trail gives |Doppler| <= {LIM:g} Hz only if its direction is within +-{np.degrees(np.arcsin(k)):.2f} deg "
          f"of the 'tangent plane' (the surface of the blue shell in the 3D view).")
    print(f"That is about {k*100:.1f}% of randomly oriented trails.")


def show_static_plots():
    """Left: zero-Doppler line and +-LIM band for several reflection points. Right: Doppler map at the midpoint."""
    az = np.arange(0, 360, 5.0)     # headings evaluated, every 5 deg
    # name -> (fraction of the way from TX to RX, altitude in m) for each reflection-point case
    alt = ALT_KM * 1e3
    cases = {f"midpoint {ALT_KM:g} km": (0.5, alt), f"{ALT_KM - 20:g} km alt": (0.5, alt - 20e3),
             f"{ALT_KM + 20:g} km alt": (0.5, alt + 20e3),
             "1/4 from TX": (0.25, alt), "1/4 from RX": (0.75, alt)}
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    ax = axs[0]                     # left panel: zero-Doppler curves for each case
    for (name, (fr, alt_m)), col in zip(cases.items(), ["k", "tab:blue", "tab:green", "tab:red", "tab:orange"]):
        P = point_on_path(fr, alt_m)
        zd = zero_doppler_desc(P, az)    # descent angle giving 0 Hz at each heading
        bw = band_halfwidth(P, az)       # (lo, hi) descent range giving |f| <= LIM at each heading
        ax.plot(az, zd, color=col, label=name)                              # zero-Doppler line
        ax.fill_between(az, bw[:, 0], bw[:, 1], color=col, alpha=0.12)      # shaded +-LIM band
    ax.set_xlim(0, 360); ax.set_ylim(0, 90); ax.set_xticks(range(0, 361, 45))
    ax.set_xlabel("meteor heading, deg (direction of travel, from N)")
    ax.set_ylabel("descent angle below horizontal, deg")
    ax.set_title(f"Zero-Doppler line (shaded: |f| <= {LIM:.0f} Hz), {F/1e6:.2f} MHz, {V_KMS:g} km/s")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axs[1]                     # right panel: Doppler map over all headings/descents at the midpoint
    P = point_on_path(0.5, ALT_KM * 1e3)
    A_, D_ = np.meshgrid(np.arange(0, 360.5, 1), np.arange(0, 90.5, 1))   # heading / descent grids (1 deg)
    Fd = doppler(P, A_, D_)         # Doppler (Hz) at every grid point
    cs = ax.pcolormesh(A_, D_, Fd, cmap="RdBu_r", vmin=-2000, vmax=2000, shading="auto")
    ax.contour(A_, D_, Fd, levels=[-LIM, LIM], colors="k", linewidths=1.2)
    ax.contour(A_, D_, Fd, levels=[0], colors="k", linewidths=0.8, linestyles="--")
    plt.colorbar(cs, ax=ax, label="Doppler, Hz (+ = path shortening)")
    ax.set_xlabel("heading, deg"); ax.set_ylabel("descent angle, deg")
    ax.set_title(f"Doppler over all headings/descents at the midpoint, {ALT_KM:g} km (black = ±{LIM:g} Hz)")
    ax.set_xticks(range(0, 361, 45))
    fig.tight_layout()
    plt.show()


def print_tables():
    """Heading -> descent tables at the midpoint, and for reflection points off the baseline."""
    az = np.arange(0, 360, 5.0)
    P0 = point_on_path(0.5, ALT_KM * 1e3)
    print(f"Midpoint, {ALT_KM:g} km altitude:  heading -> descent angle for 0 Hz, and range for +-{LIM:g} Hz")
    zd = zero_doppler_desc(P0, az); bw = band_halfwidth(P0, az)
    for a, z, (lo, hi) in zip(az[::6], zd[::6], bw[::6]):
        print(f"  {a:5.0f}°  zero at {z:5.1f}°   ±{LIM:g} Hz: {lo:5.1f}–{hi:5.1f}°")

    print()
    print(f"Reflection point off the baseline (alt {ALT_KM:g} km, midpoint along path). y>0 = left of RX->TX, y<0 = right")
    print(f"  y_km  desc  heading(0 Hz)  +-{LIM:g} Hz window   (two mirror-image solutions per row)")
    for y in (-200, -150, -100, -50, 50, 100, 150, 200):
        P = P_offset(0, y, ALT_KM * 1e3)
        for dsc in (20, 30, 45, 60):
            w = heading_windows(P, dsc)
            if np.isnan(w[0][0]):
                continue
            txt = "  |  ".join(f"{a0:5.1f}  ({lo:5.1f}..{hi:5.1f})" for a0, lo, hi in w)
            print(f"  {y:5d} {dsc:4d}   {txt}")


def scale_page(page, base_km, tx_name, rx_name, left_dir):
    """Rewrite the site-dependent numbers and labels in the page template for this baseline (km).

    The template was written for a 210 km baseline; every number is scaled by base_km / 210.
    Each replacement must find its text in the template, otherwise an error is raised, so if you
    edit the template, update the matching 'old' text below."""
    half = int(round(base_km / 2))                 # along-path slider runs +-half the baseline
    quarter = int(round(base_km / 4))              # preset / starting position: 1/4 of the baseline
    step_x = max(1, int(round(base_km / 210)))     # along-path slider step (km)
    ymax = int(round(base_km * 1.2 / 10)) * 10     # offset slider range, slightly more than the baseline
    step_y = max(1, int(round(ymax / 50)))         # offset slider step (km), 50 steps per side
    clean = lambda s, default: "".join(ch for ch in s if ch not in "'\"<>&\\") or default   # keep names HTML/JS safe
    name, rx = clean(tx_name, "TX"), clean(rx_name, "RX")
    swaps = [
        ('id="x" type="range" min="-105" max="105" step="1" value="-52"',
         f'id="x" type="range" min="{-half}" max="{half}" step="{step_x}" value="{-quarter}"'),
        ('id="y" type="range" min="-250" max="250" step="5" value="0"',
         f'id="y" type="range" min="{-ymax}" max="{ymax}" step="{step_y}" value="0"'),
        ('{q:-52,m:0,t:52}', f'{{q:{-quarter},m:0,t:{quarter}}}'),
        ('WDPN 54.31 MHz → RX forward scatter', f'{name} {F/1e6:.2f} MHz → {rx} forward scatter'),
        ("text:['WDPN TX']", f"text:['{name} TX']"),
        ("text:['RX']", f"text:['{rx} RX']"),
        ('const ok=Math.abs(f)<=100,L=60;', f'const ok=Math.abs(f)<={LIM:g},L=60;'),
        ("y0:-100,y1:100,", f"y0:-{LIM:g},y1:{LIM:g},"),
        ('|f<sub>D</sub>| ≤ 100 Hz at P', f'|f<sub>D</sub>| ≤ {LIM:g} Hz at P'),
        ('Offset from path (+ = NW)', f'Offset from path (+ = {left_dir})'),
    ]
    for old, new in swaps:
        if old not in page:
            raise ValueError(f"scale_page: could not find this text in the template: {old}")
        page = page.replace(old, new)
    print(f"page scaled for a {base_km:.0f} km baseline: along-path +-{half} km (presets +-{quarter}), "
          f"offset +-{ymax} km (+ = {left_dir})")
    return page


def _tile(z, x, y):
    """One map tile as an RGB array, from TILE_CACHE if it was downloaded before."""
    from PIL import Image
    f = TILE_CACHE / f"{z}_{x}_{y}.png"
    if not f.exists():
        req = urllib.request.Request(MAP_TILES.format(z=z, x=x, y=y), headers={"User-Agent": "meteor-scatter-3d-notebook"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read()
        TILE_CACHE.mkdir(exist_ok=True)
        f.write_bytes(data)
    return np.asarray(Image.open(io.BytesIO(f.read_bytes())).convert("RGB"))


def ground_map(lat, lon, cell_km):
    """Map colours for the ground cells: lat/lon (deg) arrays of shape (n, s, n, s), s sample points across each
    cell, are looked up in a mosaic of web-Mercator tiles and averaged per cell. Returns a hex string (6 digits
    per cell, row by row), or None if MAP_TILES is None or the tiles cannot be fetched."""
    if MAP_TILES is None:
        return None
    s = lat.shape[1]
    lat = np.clip(lat, -85, 85)
    # zoom: tile pixels a little smaller than the sample spacing, limited to about 100 tiles
    z = int(np.clip(np.ceil(np.log2(156543.03 * np.cos(np.radians(lat.mean())) / (cell_km * 1e3 / s))), 2, 12))
    while True:
        # global pixel coordinates (256-pixel tiles) of every sample point at zoom z
        px = (lon + 180) / 360 * 256 * 2 ** z
        py = (1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / np.pi) / 2 * 256 * 2 ** z
        tx0, tx1, ty0, ty1 = int(px.min() // 256), int(px.max() // 256), int(py.min() // 256), int(py.max() // 256)
        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) <= 100 or z == 2:
            break
        z -= 1
    try:
        rows = [np.concatenate([_tile(z, tx % 2 ** z, ty) for tx in range(tx0, tx1 + 1)], axis=1)
                for ty in range(ty0, ty1 + 1)]
    except Exception as err:
        print(f"map tiles unavailable ({err}); drawing plain ground")
        return None
    mosaic = np.concatenate(rows, axis=0)
    ix = np.clip((px - tx0 * 256).astype(int), 0, mosaic.shape[1] - 1)
    iy = np.clip((py - ty0 * 256).astype(int), 0, mosaic.shape[0] - 1)
    rgb = mosaic[iy, ix].astype(float).mean(axis=(1, 3)).round().astype(np.uint8)   # (n, n, 3) cell averages
    print(f"map: {(tx1 - tx0 + 1) * (ty1 - ty0 + 1)} tiles at zoom {z} ({MAP_CREDIT})")
    return rgb.tobytes().hex()


def build_interactive(tx_name, rx_name="RX", out_file=PAGE_FILE):
    """Write the interactive page for the current TX/RX and return an IFrame showing it."""
    T, R = ecef(*TX), ecef(*RX)                   # TX and RX positions in ECEF (m)
    E0, N0, U0 = enu(RX[0], RX[1])                # local East/North/Up axes at RX (the page's x/y/z)
    O = ecef(RX[0], RX[1], 0)                     # origin of the page frame (RX ground point), in ECEF
    # loc: ECEF position (m) -> page-frame [x, y, z] in km; ldir: ECEF direction -> page-frame components
    loc = lambda p: [float(((np.asarray(p) - O) / 1e3) @ a) for a in (E0, N0, U0)]
    ldir = lambda v: [float(np.asarray(v) @ a) for a in (E0, N0, U0)]
    CEN = loc([0, 0, 0])                          # Earth's center in the page frame (km)
    RE = float(np.linalg.norm(CEN))               # local Earth radius (km)
    M = loc(ecef(*point_on_path(0.5, 0)))         # baseline midpoint on the ground

    # ground patch: +-G km East/North of RX in GN x GN cells, on the sphere of radius RE about CEN (as drawn by the page)
    base = baseline_km()
    G, GN, S = float(round(1.6 * base)), 300, 3   # S x S map samples are averaged per cell
    u = (np.arange(GN * S) + 0.5) / (GN * S) * 2 * G - G
    x, y = np.meshgrid(u, u)                      # rows run South -> North, like the page's grid
    zg = CEN[2] + np.sqrt(RE ** 2 - (x - CEN[0]) ** 2 - (y - CEN[1]) ** 2)
    p = O[:, None, None] + 1e3 * (x * E0[:, None, None] + y * N0[:, None, None] + zg * U0[:, None, None])
    lat, lon, _ = geodetic(p, 0)
    sh = (GN, S, GN, S)
    MAP = ground_map(lat.reshape(sh), lon.reshape(sh), 2 * G / GN)
    zmin = CEN[2] + np.sqrt(RE ** 2 - (G + abs(CEN[0])) ** 2 - (G + abs(CEN[1])) ** 2)   # lowest ground corner
    ZR = [float(np.floor(zmin - 2)), 140.0]
    AR = dict(x=1, y=1, z=(ZR[1] - ZR[0]) / (2 * G))   # true-scale box shape

    # Constants handed to the page's JS: T/R/M positions, C Earth center, Z polar axis, RE local radius (km), LAM wavelength (m),
    # ground patch half-size G and cells GN, axis range ZR and aspect AR, map colours MAP and credit
    K = dict(T=loc(T), R=loc(R), M=M, C=CEN, Z=ldir([0, 0, 1]), RE=RE, LAM=float(LAM),
             G=G, GN=GN, ZR=ZR, AR=AR, MAP=MAP, CREDIT=MAP_CREDIT)

    # which compass direction is to the LEFT when looking from RX toward TX (the page's "+ offset" side)
    d = np.array(K["T"]) - np.array(K["R"])
    left_dir = compass(np.degrees(np.arctan2(d[0], d[1])) - 90)

    page = scale_page(PAGE, base, tx_name, rx_name, left_dir)
    html = page.replace("__PLOTLY__", get_plotlyjs()).replace("__K__", json.dumps(K))
    with open(out_file, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("wrote", out_file, round(len(html) / 1e6, 1), "MB")
    return IFrame(out_file, width="100%", height=860)
