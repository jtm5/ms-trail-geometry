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
