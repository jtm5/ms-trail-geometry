"""PROTOTYPE: polarization and trail-aspect terms for the relative strength map. Delete this file (and Step 6c of the
notebook, and sanity_check_polarization.py) to cut it out; nothing else depends on it.

meteor_detect.strength_map gives the range term only. For an underdense trail the standard result (McKinley 1961, Ceplecha
et al. 1998) has two more geometry terms, both computed here from vectors alone, with isotropic antennas:

1. Polarization. The free electrons in the trail are driven along the TX E-field and re-radiate. With a horizontally
   polarized TX and a horizontally polarized RX (and no Faraday rotation) the power factor is (e_in . e_rx)^2, where
   e_in is the horizontal direction at the TX perpendicular to the ray TX->P, and e_rx is the horizontal direction at the
   RX perpendicular to the ray P->RX. It is 1 when the two directions line up and 0 (a null) when they are at 90 degrees.
   It is floored at pol_floor_db so a null does not become minus infinity.

2. Trail aspect (Fresnel zone length). A trail scatters coherently over a length that grows as the trail lines up with the
   ray: along the trail the path length changes as (s^2 / 2) (1 - c^2) (1/R_TX + 1/R_RX), with c the cosine of the angle
   between the trail and the ray (equal for TX and RX at the specular point). The echo power therefore goes as
   1 / (1 - c^2), which is 1 (0 dB) for a trail square to the rays and grows as it lines up. Capped at aspect_cap_db
   because a real trail is not infinitely long.

All values are in dB and relative: range term as in strength_map (0 dB at a specular point 100 km from both stations), the
other two relative to a perpendicular trail seen by aligned polarizations (0 dB). Nothing here knows about noise, gain or
antenna patterns."""
import math

import numpy as np

import meteor_detect as md
from meteor_core import ecef, enu


def strength_terms(vhat, tx, rx, lat_range, lon_range, n=121, ref_alt_km=100.0, band_km=md.BAND_KM, ref_range_km=100.0,
                   pol_floor_db=-30.0, aspect_cap_db=10.0):
    """Same trail grid and pass test as md.strength_map, with the three terms kept apart.

    Returns a dict: lat, lon (the trail grid), ok (passing trails), range_db, pol_db, aspect_db (all nan where not ok),
    spec_lat, spec_lon (the specular point of every trail, deg)."""
    lat, lon = np.meshgrid(np.linspace(*lat_range, n), np.linspace(*lon_range, n), indexing="ij")
    p0 = np.stack([ecef(la, lo, ref_alt_km * 1e3) for la, lo in zip(lat.ravel(), lon.ravel())]).reshape(lat.shape + (3,))
    T, R = ecef(*tx), ecef(*rx)
    up_tx, up_rx = enu(tx[0], tx[1])[2], enu(rx[0], rx[1])[2]
    p = p0 + md.tangent_param(p0, vhat, T, R)[..., None] * vhat
    la, lo, h = md.ecef_to_geodetic(p)
    dT, dR = np.linalg.norm(p - T, axis=-1), np.linalg.norm(p - R, axis=-1)
    uT, uR = (p - T) / dT[..., None], (p - R) / dR[..., None]
    up_p = np.stack([np.cos(np.radians(la)) * np.cos(np.radians(lo)), np.cos(np.radians(la)) * np.sin(np.radians(lo)),
                     np.sin(np.radians(la))], axis=-1)
    ok = (h / 1e3 >= band_km[0]) & (h / 1e3 <= band_km[1]) & ((uT @ up_tx) >= 0) & ((uR @ up_rx) >= 0) & ((up_p @ vhat) < 0)
    rt, rr = dT / 1e3, dR / 1e3
    range_db = -10 * np.log10(rt * rr * (rt + rr) / (2 * ref_range_km ** 3))

    # polarization: horizontal E at each station, perpendicular to its ray
    e_in = np.cross(up_tx, uT)
    e_rx = np.cross(up_rx, uR)
    with np.errstate(invalid="ignore", divide="ignore"):
        e_in /= np.linalg.norm(e_in, axis=-1, keepdims=True)
        e_rx /= np.linalg.norm(e_rx, axis=-1, keepdims=True)
        pol = np.sum(e_in * e_rx, axis=-1) ** 2
        pol_db = np.where(np.isfinite(pol), 10 * np.log10(np.maximum(pol, 1e-30)), pol_floor_db)
    pol_db = np.maximum(pol_db, pol_floor_db)

    # aspect: 1 / (1 - c^2), c = cos(angle between trail and the ray)
    c = uT @ vhat
    aspect_db = np.minimum(-10 * np.log10(np.maximum(1 - c ** 2, 1e-12)), aspect_cap_db)

    nan = lambda x: np.where(ok, x, np.nan)
    return dict(lat=lat, lon=lon, ok=ok, range_db=nan(range_db), pol_db=nan(pol_db), aspect_db=nan(aspect_db),
                spec_lat=la, spec_lon=lo)


def offset_profiles(vhat, tx, rx, half_baselines=3.0, n=161, bin_km=25.0, band_km=md.BAND_KM, **kw):
    """Strongest passing trail against the specular point's distance from the TX/RX midpoint across the trail heading
    (the same offset as md.strength_profile), for each combination of terms. All curves are relative to the strongest
    range-only trail (0 dB), so the loss from the extra terms shows. Returns a dict: offset_km, range, range_pol,
    range_pol_aspect (best dB per offset bin), pol (the polarization term alone, best in bin), tx_offset_km, rx_offset_km."""
    T, R = ecef(*tx), ecef(*rx)
    mid_lat, mid_lon, _ = (float(x) for x in md.ecef_to_geodetic((T + R) / 2))
    lat_r, lon_r = md.grid_around(tx, rx, half_baselines)
    s = strength_terms(vhat, tx, rx, lat_r, lon_r, n=n, band_km=band_km, **kw)
    ok = s["ok"]
    heading = math.radians(md.heading_descent(vhat, mid_lat, mid_lon)[0])
    across = lambda la_, lo_: ((np.asarray(lo_) - mid_lon) * 111.2 * math.cos(math.radians(mid_lat))) * math.cos(heading) \
        - ((np.asarray(la_) - mid_lat) * 111.2) * math.sin(heading)
    out = dict(tx_offset_km=float(across(tx[0], tx[1])), rx_offset_km=float(across(rx[0], rx[1])))
    if not ok.any():
        return dict(out, offset_km=np.array([]), range=np.array([]), range_pol=np.array([]), range_pol_aspect=np.array([]),
                    pol=np.array([]))
    off = across(s["spec_lat"], s["spec_lon"])[ok]
    r, pl, asp = s["range_db"][ok], s["pol_db"][ok], s["aspect_db"][ok]
    ref = r.max()
    curves = dict(range=r - ref, range_pol=r + pl - ref, range_pol_aspect=r + pl + asp - ref, pol=pl)
    edges = np.arange(np.floor(off.min() / bin_km) * bin_km, off.max() + bin_km, bin_km)
    idx = np.digitize(off, edges) - 1
    cen, best = [], {k: [] for k in curves}
    for k in range(len(edges) - 1):
        sel = idx == k
        if sel.any():
            cen.append(0.5 * (edges[k] + edges[k + 1]))
            for name, v in curves.items():
                best[name].append(v[sel].max())
    return dict(out, offset_km=np.array(cen), **{k: np.array(v) for k, v in best.items()})


def explain_figure(baseline_km=210.0, p_east_km=20.0, p_north_km=150.0, alt_km=100.0, descent_deg=45.0, arrow_km=45.0,
                   aspect_demo_deg=30.0):
    """A schematic of the geometry behind the two extra terms, for one made-up specular trail on a flat Earth.

    TX and RX sit on the ground baseline_km apart (TX west, RX east). The specular point P is p_east_km east and
    p_north_km north of the middle of the baseline, alt_km up. The trail is the downward line through P at descent_deg
    whose heading satisfies the specular condition (it makes the same angle with the ray from the TX and the ray to the RX).
    Left: 3D view with the incident ray, the scattered ray and the horizontal E-field directions. Middle: the same seen from
    above, with the angle between the two E directions that sets the polarization term. Right: why the trail aspect term
    grows as the trail lines up with the rays (its lower sketch uses aspect_demo_deg, not this trail's own angle, because
    this trail's angle gives only a small effect; the real value is in the caption). The numbers come from the same
    formulas as strength_terms().
    Returns (fig, info) with info a dict of the angles and terms shown."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    unit = lambda v: v / np.linalg.norm(v)
    B = baseline_km
    T, R = np.array([-B / 2, 0.0, 0.0]), np.array([B / 2, 0.0, 0.0])
    P = np.array([p_east_km, p_north_km, alt_km])
    up = np.array([0.0, 0.0, 1.0])
    ki, ks = unit(P - T), unit(R - P)                   # ray from the TX to P, ray from P to the RX
    n = ks - ki                                          # specular: the trail is square to this vector
    d = math.radians(descent_deg)
    rn, alpha = math.hypot(n[0], n[1]), math.atan2(n[1], n[0])
    x = n[2] * math.tan(d) / rn                          # n_e sin(h) + n_n cos(h) = n_u tan(d), h = heading
    if abs(x) > 1:
        raise ValueError("No specular trail has that descent at this point; try a smaller descent_deg.")
    h = math.asin(x) - alpha
    t = np.array([math.sin(h) * math.cos(d), math.cos(h) * math.cos(d), -math.sin(d)])   # trail direction, pointing down
    c = float(ki @ t)
    theta = math.degrees(math.acos(abs(c)))              # angle between the trail and either ray
    aspect_db = -10 * math.log10(1 - c ** 2)
    e_in, e_rx = unit(np.cross(up, ki)), unit(np.cross(up, ks))
    psi = math.degrees(math.acos(min(1.0, abs(float(e_in @ e_rx)))))
    pol_db = 10 * math.log10(max(float(e_in @ e_rx) ** 2, 1e-30))
    es_amp = float(np.linalg.norm(e_in - float(e_in @ ks) * ks))
    info = dict(theta_deg=theta, aspect_db=aspect_db, psi_deg=psi, pol_db=pol_db, scattered_amp=es_amp,
                accepted_amp=abs(float(e_in @ e_rx)),
                range_tx_km=float(np.linalg.norm(P - T)), range_rx_km=float(np.linalg.norm(R - P)),
                trail_heading_deg=math.degrees(math.atan2(t[0], t[1])) % 360, descent_deg=descent_deg)

    C_IN, C_SC, C_TR, C_EI, C_ER, C_ES = "#1f6fb5", "#2a9d4b", "#8a5a14", "#e07b00", "#8a3ca8", "#0a5c2a"
    fig = plt.figure(figsize=(18, 7.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 0.8], wspace=0.08, left=0.01, right=0.99, top=0.88, bottom=0.07)
    half = 55.0

    # ---- left: 3D ----
    a3 = fig.add_subplot(gs[0], projection="3d")
    xl, yl = (-B / 2 - 35, B / 2 + 35), (-40, max(p_north_km, 60) + 50)
    ground = [[(xl[0], yl[0], 0), (xl[1], yl[0], 0), (xl[1], yl[1], 0), (xl[0], yl[1], 0)]]
    a3.add_collection3d(Poly3DCollection(ground, facecolors="#cfe8d5", alpha=0.35, edgecolors="none"))
    a3.plot(*zip(T, R), "k--", lw=1)
    a3.plot(*zip(P, [P[0], P[1], 0]), color="gray", ls=":", lw=1)
    a3.plot([P[0]], [P[1]], [0], marker="x", color="gray", ms=6)
    for pt, col, lab in ((T, "red", "TX"), (R, "blue", "RX")):
        a3.plot([pt[0]], [pt[1]], [pt[2]], "^", color=col, mec="k", ms=10)
        a3.text(pt[0], pt[1] - 22, pt[2], lab, fontsize=11, ha="center", weight="bold")
    a3.plot(*zip(P - t * half, P + t * half), color=C_TR, lw=5, solid_capstyle="round")
    a3.plot([P[0]], [P[1]], [P[2]], "o", color="k", ms=6)
    a3.quiver(*T, *(P - T), arrow_length_ratio=0.04, color=C_IN, lw=2)
    a3.quiver(*P, *(R - P), arrow_length_ratio=0.04, color=C_SC, lw=2)
    for base in (T, P):
        a3.quiver(*base, *(e_in * arrow_km), arrow_length_ratio=0.25, color=C_EI, lw=2.5)
    # the scattered E: the driven (horizontal) direction with its along-the-ray part removed, so it tilts out of the horizontal.
    # It keeps that direction all the way to the RX; the RX accepts only its horizontal part.
    es = e_in - float(e_in @ ks) * ks
    if es[2] < 0:
        es = -es                                          # the field oscillates both ways; draw it pointing up
    got = float(es @ e_rx)                              # signed amplitude the horizontal RX accepts (= e_in . e_rx)
    for f in (0.25, 0.5, 0.75, 1.0):
        a3.quiver(*(P + f * (R - P)), *(es * arrow_km), arrow_length_ratio=0.25, color=C_ES, lw=2.2)
    a3.quiver(*R, *(e_rx * arrow_km), arrow_length_ratio=0.25, color=C_ER, lw=1.2, alpha=0.35)
    a3.quiver(*R, *(e_rx * got * arrow_km), arrow_length_ratio=0.3, color=C_ER, lw=3.2)
    a3.plot(*zip(R + es * arrow_km, R + e_rx * got * arrow_km), color="k", ls=":", lw=1)
    a3.text(*((T + P) / 2 + [-30, 0, 8]), "incident wave\nTX to P", color=C_IN, fontsize=9, ha="center")
    a3.text(*((P + R) / 2 + [-8, 0, 2]), "scattered wave\nP to RX", color=C_SC, fontsize=9, ha="right")
    a3.text(*(P + [-18, 0, 14]), "specular point P\n(on the trail)", fontsize=9, ha="right")
    a3.text(*(P - t * half + [14, 0, 4]), "meteor trail", color=C_TR, fontsize=9, ha="left")
    a3.text(*(T + e_in * arrow_km + [0, 0, 8]), "E sent by TX\n(horizontal)", color=C_EI, fontsize=9, ha="center")
    a3.text(*(R + e_rx * arrow_km + [14, 0, 6]), "E the RX accepts: only the\nhorizontal part of the\nscattered E (dotted drop)", color=C_ER, fontsize=9, ha="left")
    a3.text(*(P + 0.5 * (R - P) + [30, 0, -4]), "E of the scattered wave:\nsquare to its ray, so tilted\nout of the horizontal, and it\nstays that way all the way\nto the RX", color=C_ES, fontsize=9, ha="left")
    a3.set_xlim(xl); a3.set_ylim(yl); a3.set_zlim(0, alt_km * 1.4)
    a3.set_box_aspect((xl[1] - xl[0], yl[1] - yl[0], (alt_km * 1.4) * 1.6))
    a3.set_xticks([]); a3.set_yticks([]); a3.set_zticks([0, alt_km])
    a3.set_zticklabels(["ground", f"{alt_km:g} km"], fontsize=8)
    a3.view_init(elev=22, azim=-62)
    a3.set_title("the geometry (flat Earth, not to a real scale)", fontsize=11)

    # ---- middle: plan view ----
    ap = fig.add_subplot(gs[1])
    ap.plot(*zip(T[:2], R[:2]), "k--", lw=1)
    ap.plot(*zip((P - t * half)[:2], (P + t * half)[:2]), color=C_TR, lw=5, solid_capstyle="round", alpha=0.8)
    for pt, col, lab in ((T, "red", "TX"), (R, "blue", "RX")):
        ap.plot(pt[0], pt[1], "^", color=col, mec="k", ms=11)
        ap.annotate(lab, pt[:2], textcoords="offset points", xytext=(0, -16), ha="center", fontsize=10)
    ap.plot(P[0], P[1], "o", color="k", ms=6)
    ap.annotate("P", P[:2], textcoords="offset points", xytext=(-12, -14), fontsize=10, weight="bold")
    ap.annotate("", xy=P[:2], xytext=T[:2], arrowprops=dict(arrowstyle="-|>", color=C_IN, lw=2))
    ap.annotate("", xy=R[:2], xytext=P[:2], arrowprops=dict(arrowstyle="-|>", color=C_SC, lw=2))
    for base, e, col in ((T, e_in, C_EI), (P, e_in, C_EI), (R, e_rx, C_ER)):
        ap.annotate("", xy=(base + e * arrow_km)[:2], xytext=base[:2], arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
    ap.set_aspect("equal"); ap.set_xlim(xl); ap.set_ylim(yl[0] - 10, yl[1]); ap.set_xticks([]); ap.set_yticks([])
    ap.set_title(f"the same, seen from above\nPolarization term for this trail: {pol_db:.1f} dB", fontsize=10.5)
    ap.set_xlabel("Each E arrow is horizontal and square to its ray, so it is fixed by that ray's compass direction.\n"
                  "Where the two rays point in different directions the two E arrows do not line up.", fontsize=8.5)
    ins = ap.inset_axes([0.04, 0.50, 0.28, 0.28])
    for v, col in ((e_in, C_EI), (e_rx, C_ER)):
        ins.annotate("", xy=v[:2], xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
    a0, a1 = math.degrees(math.atan2(e_in[1], e_in[0])), math.degrees(math.atan2(e_rx[1], e_rx[0]))
    if abs(a0 - a1) > 180:
        a1 += 360 if a1 < a0 else -360
    arc = np.radians(np.linspace(a0, a1, 30))
    ins.plot(0.45 * np.cos(arc), 0.45 * np.sin(arc), color="k", lw=1)
    ins.set_xlim(-1.1, 1.1); ins.set_ylim(-1.1, 1.1); ins.set_aspect("equal"); ins.axis("off")
    ins.set_title(f"angle between the two E arrows\n{psi:.0f} deg gives {pol_db:.1f} dB", fontsize=8.5)

    # ---- right: aspect ----
    sub = gs[2].subgridspec(2, 1, hspace=0.35)
    demo_db = -10 * math.log10(math.sin(math.radians(aspect_demo_deg)) ** 2)
    cases = ((90.0, "trail square to the rays: 0 dB", "short stretch of trail adds in phase"),
             (aspect_demo_deg, f"trail lined up with the rays ({aspect_demo_deg:.0f} deg): {demo_db:+.1f} dB", "longer stretch adds in phase"))
    for k, (th, title, note) in enumerate(cases):
        ax = fig.add_subplot(sub[k])
        r = math.radians(th)
        zone = min(0.55 / math.sin(r), 1.0)           # coherent length along the trail grows as 1 / sin(angle)
        ax.plot([0, 0], [-1.1, 1.1], color=C_TR, lw=2, alpha=0.5)
        ax.plot([0, 0], [-zone / 2, zone / 2], color=C_TR, lw=7, solid_capstyle="butt")
        left, right = np.array([-math.sin(r), -math.cos(r)]) * 1.2, np.array([math.sin(r), -math.cos(r)]) * 1.2
        ax.annotate("", xy=(0, 0), xytext=left, arrowprops=dict(arrowstyle="-|>", color=C_IN, lw=2))
        ax.annotate("", xy=right, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C_SC, lw=2))
        ax.text(0.08, zone / 2 + 0.03, "coherent length", fontsize=8.5, color=C_TR, va="bottom")
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title + "\n" + note, fontsize=9.5)
    fig.text(0.86, 0.03, f"Aspect term: 1 / sin^2 of the angle between trail and rays.\nThis example trail: {theta:.0f} deg, {aspect_db:+.1f} dB.",
             fontsize=9, ha="center")

    fig.suptitle("Where the two Step 6c terms come from.  Polarization: cos^2 of the angle between the orange (sent) and purple (accepted) E arrows.  "
                 "Aspect: 1 / sin^2 of the angle between the trail and the rays.", fontsize=11)
    return fig, info
