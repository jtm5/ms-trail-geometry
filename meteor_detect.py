"""Can a meteor trail from a given radiant give a specular (trail-echo) detection for a TX/RX pair?

Geometry only. No echo power, no link budget (see NOTES_specular_detection_project.md). The trail is a straight
line. It reflects specularly where it is tangent to the TX/RX path-length ellipsoid, which is the point on the line
that minimises |P - TX| + |P - RX|. "Possible" means that point lies in the ionised height band, is above the
horizon of both stations, and the trail is heading downward there (the radiant is above the horizon at that place).

All vectors are ECEF (m) unless a name says otherwise. Sites are (lat deg, lon deg, height m) like meteor_core.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from meteor_core import A, E2, ecef, enu

# ---- Named assumptions (inputs, not buried in the maths) --------------------------------------
BAND_KM = (70.0, 120.0)    # ionised height band for a trail echo
TOL_DEG = 0.5              # trail-direction tolerance around the nominal direction (about 100 Hz at 54 MHz, 20 km/s)
SEARCH_M = 5.0e6           # the tangent point is searched for within +-this distance along the line


# ---- Radiant, time and place -> trail direction -----------------------------------------------
def julian_date(when):
    """Julian date of a datetime (naive values are taken as UTC)."""
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc).replace(tzinfo=None)
    y, m = when.year, when.month
    if m <= 2:
        y, m = y - 1, m + 12
    a = y // 100
    b = 2 - a + a // 4
    day = when.day + (when.hour + (when.minute + (when.second + when.microsecond / 1e6) / 60) / 60) / 24
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + day + b - 1524.5


def gmst_deg(when):
    """Greenwich mean sidereal time (deg)."""
    jd = julian_date(when)
    t = (jd - 2451545.0) / 36525
    return (280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * t ** 2 - t ** 3 / 38710000) % 360


def precess(ra_deg, dec_deg, when):
    """Move a J2000 RA/Dec to the equator and equinox of the date (Meeus low-order precession), in degrees.

    Over 26 years the shift is about 0.36 deg, comparable to the tolerance, so it is not worth skipping."""
    t = (julian_date(when) - 2451545.0) / 36525
    zeta = math.radians(0.6406161 * t + 0.0000839 * t ** 2)
    z = math.radians(0.6406161 * t + 0.0003041 * t ** 2)
    theta = math.radians(0.5567530 * t - 0.0001185 * t ** 2)
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    a = math.cos(dec) * math.sin(ra + zeta)
    b = math.cos(theta) * math.cos(dec) * math.cos(ra + zeta) - math.sin(theta) * math.sin(dec)
    c = math.sin(theta) * math.cos(dec) * math.cos(ra + zeta) + math.cos(theta) * math.sin(dec)
    return math.degrees(math.atan2(a, b) + z) % 360, math.degrees(math.asin(c))


def trail_direction(ra_deg, dec_deg, when, j2000=True):
    """Unit ECEF vector of the direction the meteors travel (away from the radiant) at this time.

    Parallel trails: every meteor from the radiant has this same direction at every place on Earth.
    Ignored for now: zenith attraction, Earth's rotation velocity, atmospheric deceleration."""
    if j2000:
        ra_deg, dec_deg = precess(ra_deg, dec_deg, when)
    lon = math.radians(ra_deg - gmst_deg(when))     # hour-angle-like longitude of the radiant direction
    dec = math.radians(dec_deg)
    toward = np.array([math.cos(dec) * math.cos(lon), math.cos(dec) * math.sin(lon), math.sin(dec)])
    return -toward


def heading_descent(vhat, lat, lon):
    """(heading deg from N, descent deg below horizontal) of a direction at a place. Same convention as meteor_core.doppler."""
    e, n, u = enu(lat, lon)
    return math.degrees(math.atan2(vhat @ e, vhat @ n)) % 360, math.degrees(math.asin(-(vhat @ u)))


# ---- Geodetic altitude --------------------------------------------------------------------------
def ecef_to_geodetic(p):
    """(lat deg, lon deg, height m) of ECEF points p (shape (..., 3)); Bowring's iteration, WGS-84."""
    p = np.asarray(p, dtype=float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    r = np.hypot(x, y)
    lat = np.arctan2(z, r * (1 - E2))
    for _ in range(6):
        n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + E2 * n * np.sin(lat), r)
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    h = r / np.cos(lat) - n if abs(np.cos(lat)).min() > 1e-6 else abs(z) / np.abs(np.sin(lat)) - n * (1 - E2)
    return np.degrees(lat), np.degrees(np.arctan2(y, x)), h


# ---- Tangent point ------------------------------------------------------------------------------
def tangent_param(p0, vhat, tx_ecef, rx_ecef, iters=60):
    """Distance t (m) along the line p0 + t*vhat where |P-TX| + |P-RX| is smallest.

    p0 may be one point or an array (..., 3); vhat is one unit vector. The slope of the path length is
    vhat . (uT + uR), which only increases along the line, so bisection finds the unique minimum."""
    p0 = np.asarray(p0, dtype=float)
    lo = np.full(p0.shape[:-1], -SEARCH_M)
    hi = np.full(p0.shape[:-1], SEARCH_M)

    def slope(t):
        p = p0 + t[..., None] * vhat
        uT = p - tx_ecef; uT /= np.linalg.norm(uT, axis=-1, keepdims=True)
        uR = p - rx_ecef; uR /= np.linalg.norm(uR, axis=-1, keepdims=True)
        return (uT + uR) @ vhat

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        up = slope(mid) > 0           # slope positive: minimum is behind mid
        hi = np.where(up, mid, hi)
        lo = np.where(up, lo, mid)
    return 0.5 * (lo + hi)


def cone_directions(vhat, tol_deg, n=8):
    """The nominal direction plus n directions tilted by tol_deg around it (for the tolerance check)."""
    a = np.cross(vhat, [0.0, 0.0, 1.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(vhat, [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(vhat, a)
    tol = math.radians(tol_deg)
    out = [vhat]
    for k in range(n):
        phi = 2 * math.pi * k / n
        w = math.cos(tol) * vhat + math.sin(tol) * (math.cos(phi) * a + math.sin(phi) * b)
        out.append(w / np.linalg.norm(w))
    return out


# ---- One trail: full geometry plus a thin verdict -----------------------------------------------
@dataclass
class TrailResult:
    """Everything known about one trail's specular point. A verdict is built from these values, not stored instead of them."""
    p_ecef: np.ndarray          # tangent (specular) point (m)
    lat: float; lon: float; alt_km: float
    uT: np.ndarray              # unit vector TX -> P (incident direction)
    uR: np.ndarray              # unit vector RX -> P (reversed scattered direction)
    range_tx_km: float; range_rx_km: float
    scatter_angle_deg: float    # angle between incident and scattered directions (0 = straight on, 180 = back-scatter)
    elev_tx_deg: float; elev_rx_deg: float   # elevation of P seen from TX / from RX (negative = below that station's horizon)
    heading_deg: float; descent_deg: float   # trail direction at P
    in_band: bool               # tangent point inside BAND_KM
    visible: bool               # P is above the horizon of both stations
    descending: bool            # the trail goes downward at P (radiant above the horizon there)
    alt_range_km: tuple         # tangent altitude over the +-tolerance cone: (lowest, highest)
    possible: bool              # nominal direction passes
    possible_within_tol: bool   # some direction in the tolerance cone passes


def _geometry(p0, vhat, tx, rx):
    """Tangent point for one trail and the values the verdict needs (no verdict yet)."""
    T, R = ecef(*tx), ecef(*rx)
    t = float(tangent_param(p0, vhat, T, R))
    p = np.asarray(p0, dtype=float) + t * vhat
    lat, lon, h = ecef_to_geodetic(p)
    lat, lon, h = float(lat), float(lon), float(h)
    uT = (p - T) / np.linalg.norm(p - T)
    uR = (p - R) / np.linalg.norm(p - R)
    return p, lat, lon, h, T, R, uT, uR


def assess_trail(p0, vhat, tx, rx, band_km=BAND_KM, tol_deg=TOL_DEG):
    """Specular geometry and verdict for the trail through ECEF point p0 with unit direction vhat."""
    p, lat, lon, h, T, R, uT, uR = _geometry(p0, vhat, tx, rx)
    up_tx, up_rx = enu(tx[0], tx[1])[2], enu(rx[0], rx[1])[2]
    elev_tx = math.degrees(math.asin(np.clip(uT @ up_tx, -1, 1)))   # uT points from TX to P
    elev_rx = math.degrees(math.asin(np.clip(uR @ up_rx, -1, 1)))
    in_band = band_km[0] <= h / 1e3 <= band_km[1]
    visible = elev_tx >= 0 and elev_rx >= 0
    descending = bool(vhat @ enu(lat, lon)[2] < 0)
    heading, descent = heading_descent(vhat, lat, lon)
    alts, ok = [], []                       # same test for each direction in the tolerance cone
    for w in cone_directions(vhat, tol_deg):
        q, qlat, qlon, qh, _, _, quT, quR = _geometry(p0, w, tx, rx)
        alts.append(qh / 1e3)
        ok.append(band_km[0] <= qh / 1e3 <= band_km[1] and (quT @ up_tx) >= 0 and (quR @ up_rx) >= 0
                  and w @ enu(qlat, qlon)[2] < 0)
    return TrailResult(
        p_ecef=p, lat=lat, lon=lon, alt_km=h / 1e3, uT=uT, uR=uR,
        range_tx_km=float(np.linalg.norm(p - T)) / 1e3, range_rx_km=float(np.linalg.norm(p - R)) / 1e3,
        scatter_angle_deg=math.degrees(math.acos(np.clip(uT @ uR, -1, 1))),
        elev_tx_deg=elev_tx, elev_rx_deg=elev_rx, heading_deg=heading, descent_deg=descent,
        in_band=in_band, visible=visible, descending=descending, alt_range_km=(min(alts), max(alts)),
        possible=in_band and visible and descending, possible_within_tol=any(ok))


def trail_through(lat, lon, alt_m, vhat, tx, rx, **kw):
    """assess_trail() for the trail passing through the point at (lat, lon, alt_m)."""
    return assess_trail(ecef(lat, lon, alt_m), vhat, tx, rx, **kw)


# ---- Whole radiant: which trails work? ----------------------------------------------------------
def radiant_map(vhat, tx, rx, lat_range, lon_range, n=121, ref_alt_km=100.0, tol_deg=TOL_DEG, band_km=BAND_KM):
    """Detectability of the family of parallel trails with direction vhat.

    Each trail is labelled by where it crosses ref_alt_km over a lat/lon grid. Returns (lat, lon, alt_nom,
    alt_lo, alt_hi, status): tangent altitude (km) for the nominal direction and the lowest and highest over the
    tolerance cone, and status 2 = nominal passes, 1 = passes only within tolerance, 0 = fails. Passing needs the
    specular point in the band, above both station horizons, and the trail heading downward there."""
    lat, lon = np.meshgrid(np.linspace(*lat_range, n), np.linspace(*lon_range, n), indexing="ij")
    p0 = np.stack([ecef(la, lo, ref_alt_km * 1e3) for la, lo in zip(lat.ravel(), lon.ravel())]).reshape(lat.shape + (3,))
    T, R = ecef(*tx), ecef(*rx)
    up_tx, up_rx = enu(tx[0], tx[1])[2], enu(rx[0], rx[1])[2]
    alts, oks = [], []
    for w in cone_directions(vhat, tol_deg):
        p = p0 + tangent_param(p0, w, T, R)[..., None] * w
        la, lo, h = ecef_to_geodetic(p)
        uT = (p - T) / np.linalg.norm(p - T, axis=-1, keepdims=True)
        uR = (p - R) / np.linalg.norm(p - R, axis=-1, keepdims=True)
        alts.append(h / 1e3)
        up_p = np.stack([np.cos(np.radians(la)) * np.cos(np.radians(lo)), np.cos(np.radians(la)) * np.sin(np.radians(lo)),
                         np.sin(np.radians(la))], axis=-1)    # local up at each specular point
        oks.append((h / 1e3 >= band_km[0]) & (h / 1e3 <= band_km[1]) & ((uT @ up_tx) >= 0) & ((uR @ up_rx) >= 0) & ((up_p @ w) < 0))
    alts, oks = np.array(alts), np.array(oks)
    status = np.where(oks[0], 2, np.where(oks.any(axis=0), 1, 0))
    return lat, lon, alts[0], alts.min(axis=0), alts.max(axis=0), status


def strength_map(vhat, tx, rx, lat_range, lon_range, n=121, ref_alt_km=100.0, band_km=BAND_KM, ref_range_km=100.0,
                 with_points=False):
    """Relative echo strength of the trails that pass (nominal direction only), from the ranges alone.

    Same trail grid and pass test as radiant_map. For an underdense trail the received power goes roughly as
    1 / (R_TX * R_RX * (R_TX + R_RX)), R being the range to the specular point (isotropic antennas, polarization and
    scattering-angle terms left out). The value is in dB relative to a specular point 100 km (ref_range_km) from both
    stations, so it can be compared between times. Returns (lat, lon, ok, db): ok is True where the trail passes, db is
    nan where it does not. With with_points=True it also returns the specular point's latitude and longitude (deg) for every
    trail, as (lat, lon, ok, db, spec_lat, spec_lon). Relative only: it says nothing about whether any trail clears the noise."""
    lat, lon = np.meshgrid(np.linspace(*lat_range, n), np.linspace(*lon_range, n), indexing="ij")
    p0 = np.stack([ecef(la, lo, ref_alt_km * 1e3) for la, lo in zip(lat.ravel(), lon.ravel())]).reshape(lat.shape + (3,))
    T, R = ecef(*tx), ecef(*rx)
    up_tx, up_rx = enu(tx[0], tx[1])[2], enu(rx[0], rx[1])[2]
    p = p0 + tangent_param(p0, vhat, T, R)[..., None] * vhat
    la, lo, h = ecef_to_geodetic(p)
    dT, dR = np.linalg.norm(p - T, axis=-1), np.linalg.norm(p - R, axis=-1)
    uT, uR = (p - T) / dT[..., None], (p - R) / dR[..., None]
    up_p = np.stack([np.cos(np.radians(la)) * np.cos(np.radians(lo)), np.cos(np.radians(la)) * np.sin(np.radians(lo)),
                     np.sin(np.radians(la))], axis=-1)
    ok = (h / 1e3 >= band_km[0]) & (h / 1e3 <= band_km[1]) & ((uT @ up_tx) >= 0) & ((uR @ up_rx) >= 0) & ((up_p @ vhat) < 0)
    rt, rr = dT / 1e3, dR / 1e3
    db = -10 * np.log10(rt * rr * (rt + rr) / (2 * ref_range_km ** 3))
    if with_points:
        return lat, lon, ok, np.where(ok, db, np.nan), la, lo
    return lat, lon, ok, np.where(ok, db, np.nan)


def margin_scan(ra_deg, dec_deg, day, tx, rx, margins_db=(3.0, 6.0, 10.0), box_km=1500.0, step_min=30, band_km=BAND_KM, n=121):
    """The rate index of rate_scan, limited to trails that keep a link margin. For each time step of a UTC day, the pass
    area of the trails (fixed square box_km to each side of the TX/RX midpoint, nominal direction) whose relative strength
    is within each margin of the strongest trail found anywhere that day, times sin(radiant elevation). A margin of
    m dB means: count only trails at most m dB weaker than the day's best (isotropic antennas, ranges only, see
    strength_map). Returns a list of dicts: when, radiant_alt, sun_alt, index_all_km2 (no margin limit) and
    index_km2 (dict margin -> km2). Relative only."""
    T, R = ecef(*tx), ecef(*rx)
    mid_lat, mid_lon, _ = (float(x) for x in ecef_to_geodetic((T + R) / 2))
    base_km = float(np.linalg.norm(T - R)) / 1e3
    lat_r, lon_r = grid_around(tx, rx, box_km / base_km)
    cell_km2 = (2 * box_km) ** 2 / n ** 2
    rows, dbs = [], []
    for k in range(0, 24 * 60, step_min):
        when = datetime(day.year, day.month, day.day, k // 60, k % 60, tzinfo=timezone.utc)
        vhat = trail_direction(ra_deg, dec_deg, when)
        descent = heading_descent(vhat, mid_lat, mid_lon)[1]
        rows.append(dict(when=when, radiant_alt=descent, sun_alt=sun_alt_deg(when, mid_lat, mid_lon)))
        if descent > 0:
            _, _, ok, db = strength_map(vhat, tx, rx, lat_r, lon_r, n=n, band_km=band_km)
            dbs.append(db[ok])
        else:
            dbs.append(np.array([]))
    day_best = max((d.max() for d in dbs if d.size), default=0.0)
    for row, d in zip(rows, dbs):
        s = math.sin(math.radians(row["radiant_alt"])) if row["radiant_alt"] > 0 else 0.0
        row["index_all_km2"] = d.size * cell_km2 * s
        row["index_km2"] = {m: float((d >= day_best - m).sum()) * cell_km2 * s for m in margins_db}
    return rows


def strength_profile(vhat, tx, rx, half_baselines=3.0, n=161, bin_km=25.0, band_km=BAND_KM):
    """Relative strength of the passing trails against how far they sit from the TX/RX midpoint across the trail direction.

    The offset is the specular point's distance from the midpoint, horizontal and perpendicular to the trails' heading
    (the direction the band of working trails runs). Several trails share an offset (they differ along the heading), so
    each offset bin gives the strongest and the weakest. dB are relative to the strongest passing trail (0 dB). Returns a
    dict: offset_km (bin centres), best_db, worst_db, count (trails per bin), tx_offset_km, rx_offset_km, spread_db (all
    passing trails)."""
    T, R = ecef(*tx), ecef(*rx)
    mid_lat, mid_lon, _ = (float(x) for x in ecef_to_geodetic((T + R) / 2))
    lat_r, lon_r = grid_around(tx, rx, half_baselines)
    _, _, ok, db, slat, slon = strength_map(vhat, tx, rx, lat_r, lon_r, n=n, band_km=band_km, with_points=True)
    heading = math.radians(heading_descent(vhat, mid_lat, mid_lon)[0])
    east = lambda lo_, la_: (np.asarray(lo_) - mid_lon) * 111.2 * math.cos(math.radians(mid_lat))
    north = lambda la_: (np.asarray(la_) - mid_lat) * 111.2
    across = lambda la_, lo_: east(lo_, la_) * math.cos(heading) - north(la_) * math.sin(heading)
    off, d = across(slat, slon)[ok], db[ok]
    out = dict(tx_offset_km=float(across(tx[0], tx[1])), rx_offset_km=float(across(rx[0], rx[1])), spread_db=d - d.max() if d.size else d)
    if not d.size:
        return dict(out, offset_km=np.array([]), best_db=np.array([]), worst_db=np.array([]), count=np.array([]))
    d = d - d.max()
    edges = np.arange(np.floor(off.min() / bin_km) * bin_km, off.max() + bin_km, bin_km)
    idx = np.digitize(off, edges) - 1
    cen, best, worst, cnt = [], [], [], []
    for k in range(len(edges) - 1):
        sel = idx == k
        if sel.any():
            cen.append(0.5 * (edges[k] + edges[k + 1])); best.append(d[sel].max()); worst.append(d[sel].min()); cnt.append(int(sel.sum()))
    return dict(out, offset_km=np.array(cen), best_db=np.array(best), worst_db=np.array(worst), count=np.array(cnt))



# ---- Sun, for the dawn/dusk pattern --------------------------------------------------------------
def sun_alt_deg(when, lat, lon):
    """Sun elevation (deg) at lat/lon (deg) and time: low-precision solar position, good to about 0.1 deg."""
    n = julian_date(when) - 2451545.0
    mean_lon = 280.460 + 0.9856474 * n
    anomaly = math.radians(357.528 + 0.9856003 * n)
    ecl_lon = math.radians(mean_lon + 1.915 * math.sin(anomaly) + 0.020 * math.sin(2 * anomaly))
    eps = math.radians(23.439 - 0.0000004 * n)
    ra = math.degrees(math.atan2(math.cos(eps) * math.sin(ecl_lon), math.cos(ecl_lon)))
    dec = math.asin(math.sin(eps) * math.sin(ecl_lon))
    ha = math.radians(gmst_deg(when) + lon - ra)
    la = math.radians(lat)
    return math.degrees(math.asin(math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(ha)))


# ---- Grid, map and scans used by the notebook ----------------------------------------------------
def grid_around(tx, rx, half_baselines=1.2):
    """(lat_range, lon_range) in degrees: a box centred on the TX/RX midpoint, half_baselines baselines to each side."""
    T, R = ecef(*tx), ecef(*rx)
    mid = ecef_to_geodetic((T + R) / 2)
    base_km = float(np.linalg.norm(T - R)) / 1e3
    dlat = half_baselines * base_km / 111.2
    dlon = dlat / math.cos(math.radians(float(mid[0])))
    return (float(mid[0]) - dlat, float(mid[0]) + dlat), (float(mid[1]) - dlon, float(mid[1]) + dlon)


def background_image(lat_range, lon_range, n=640):
    """Map picture (n, n, 3 uint8) of the lat/lon box for imshow(origin='lower'), or None if the tiles are not available."""
    import meteor_core as mc
    s = 1
    lat, lon = np.meshgrid(np.linspace(*lat_range, n * s), np.linspace(*lon_range, n * s), indexing="ij")
    cell_km = (lat_range[1] - lat_range[0]) * 111.2 / n
    hexed = mc.ground_map(lat.reshape(n, s, n, s), lon.reshape(n, s, n, s), cell_km)
    if hexed is None:
        return None
    return np.frombuffer(bytes.fromhex(hexed), dtype=np.uint8).reshape(n, n, 3)


def plot_radiant_map(vhat, tx, rx, when, title, tol_deg=TOL_DEG, band_km=BAND_KM, n=161, ax=None, background=True,
                     half_baselines=1.2):
    """Draw the detectable footprint of the trails from one radiant over a map. half_baselines sets the map size:
    that many TX-RX baselines to each side of the midpoint (about 6 shows the whole footprint for a low radiant).
    Returns the status grid."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    lat_r, lon_r = grid_around(tx, rx, half_baselines)
    lat, lon, alt, alo, ahi, status = radiant_map(vhat, tx, rx, lat_r, lon_r, n=n, tol_deg=tol_deg, band_km=band_km)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))
    extent = (lon_r[0], lon_r[1], lat_r[0], lat_r[1])
    img = background_image(lat_r, lon_r) if background else None
    if img is not None:
        ax.imshow(img, origin="lower", extent=extent, aspect="auto", zorder=0)
    cmap = ListedColormap([(0.85, 0.15, 0.15, 0.18), (1.0, 0.75, 0.0, 0.50), (0.1, 0.65, 0.2, 0.55)])
    ax.pcolormesh(lon, lat, status, cmap=cmap, vmin=0, vmax=2, shading="auto", zorder=1)
    ax.plot([tx[1], rx[1]], [tx[0], rx[0]], "k--", lw=1, zorder=2)
    ax.plot(tx[1], tx[0], "r^", ms=11, mec="k", zorder=3, label="TX")
    ax.plot(rx[1], rx[0], "b^", ms=11, mec="k", zorder=3, label="RX")
    ax.set_xlim(lon_r); ax.set_ylim(lat_r)
    ax.set_aspect(1 / math.cos(math.radians(np.mean(lat_r))))
    ax.set_xlabel("longitude, deg"); ax.set_ylabel("latitude, deg")
    ax.set_title(title)
    ax.legend(handles=[Patch(color=(0.1, 0.65, 0.2, 0.7), label="specular point in band (nominal)"),
                       Patch(color=(1.0, 0.75, 0.0, 0.7), label=f"only within +-{tol_deg:g} deg of the trail direction"),
                       Patch(color=(0.85, 0.15, 0.15, 0.4), label="no specular point in band"),
                       plt.Line2D([], [], marker="^", color="r", mec="k", ls="", label="TX"),
                       plt.Line2D([], [], marker="^", color="b", mec="k", ls="", label="RX")],
              loc="lower left", fontsize=8, framealpha=0.9)
    return status


def hourly_scan(ra_deg, dec_deg, day, tx, rx, step_min=30, tol_deg=TOL_DEG, band_km=BAND_KM, n=61):
    """For each time step of a UTC day: radiant elevation and heading at the TX/RX midpoint, sun elevation there, and the
    share of the trail grid (grid_around) that passes, nominal and including the tolerance. Returns a list of dicts."""
    T, R = ecef(*tx), ecef(*rx)
    mid_lat, mid_lon, _ = (float(x) for x in ecef_to_geodetic((T + R) / 2))
    lat_r, lon_r = grid_around(tx, rx)
    rows = []
    for k in range(0, 24 * 60, step_min):
        when = datetime(day.year, day.month, day.day, k // 60, k % 60, tzinfo=timezone.utc)
        vhat = trail_direction(ra_deg, dec_deg, when)
        heading, descent = heading_descent(vhat, mid_lat, mid_lon)
        row = dict(when=when, radiant_alt=descent, heading=heading, sun_alt=sun_alt_deg(when, mid_lat, mid_lon),
                   nominal=0.0, with_tol=0.0)
        if descent > 0:
            status = radiant_map(vhat, tx, rx, lat_r, lon_r, n=n, tol_deg=tol_deg, band_km=band_km)[5]
            row["nominal"], row["with_tol"] = 100 * float((status == 2).mean()), 100 * float((status >= 1).mean())
        rows.append(row)
    return rows


def rate_scan(ra_deg, dec_deg, day, tx, rx, box_km=1500.0, step_min=30, tol_deg=TOL_DEG, band_km=BAND_KM, n=121):
    """A relative rate index for each time step of a UTC day. Meteors arrive uniformly over the ground, so the area of the
    trail grid that passes (a square of box_km to each side of the TX/RX midpoint, nominal direction) is proportional to the
    rate of passing trails at one moment. Between times the same flux is spread over more ground when the radiant is low, so
    the index is pass area times sin(radiant elevation). Returns a list of dicts: when, radiant_alt, sun_alt, area_km2 (pass
    area), index_km2 (pass area x sin(elevation)). Relative only: no meteor flux, brightness or echo strength."""
    T, R = ecef(*tx), ecef(*rx)
    mid_lat, mid_lon, _ = (float(x) for x in ecef_to_geodetic((T + R) / 2))
    base_km = float(np.linalg.norm(T - R)) / 1e3
    lat_r, lon_r = grid_around(tx, rx, box_km / base_km)
    rows = []
    for k in range(0, 24 * 60, step_min):
        when = datetime(day.year, day.month, day.day, k // 60, k % 60, tzinfo=timezone.utc)
        vhat = trail_direction(ra_deg, dec_deg, when)
        heading, descent = heading_descent(vhat, mid_lat, mid_lon)
        row = dict(when=when, radiant_alt=descent, sun_alt=sun_alt_deg(when, mid_lat, mid_lon), area_km2=0.0, index_km2=0.0)
        if descent > 0:
            status = radiant_map(vhat, tx, rx, lat_r, lon_r, n=n, tol_deg=tol_deg, band_km=band_km)[5]
            row["area_km2"] = float((status == 2).mean()) * (2 * box_km) ** 2
            row["index_km2"] = row["area_km2"] * math.sin(math.radians(descent))
        rows.append(row)
    return rows
