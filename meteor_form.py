"""Input form for the notebook: stations, radiant, date and time.

show_form() displays the fields; values() returns the validated inputs for the cells that follow. The form remembers what
you typed while the kernel runs, so choosing Run All again rebuilds the form with your values, not the starting ones.
The maths lives in meteor_core.py and meteor_detect.py."""
from datetime import date, datetime, timezone
from types import SimpleNamespace

import ipywidgets as w
import numpy as np
from IPython.display import display

import meteor_core as mc
import meteor_detect as md

# Approximate radiants near peak activity (J2000 RA, Dec in degrees). A radiant moves a little from night to night, so
# check an up-to-date list (for example the IMO meteor shower calendar) for the date you care about.
SHOWERS = {
    "Quadrantids": (230.0, 49.0),
    "Lyrids": (271.0, 34.0),
    "Eta Aquariids": (338.0, -1.0),
    "Daytime Arietids": (44.0, 24.0),
    "Perseids": (48.0, 58.0),
    "Orionids": (95.0, 16.0),
    "Leonids": (152.0, 22.0),
    "Geminids": (112.0, 33.0),
    "Ursids": (217.0, 76.0),
}
CUSTOM = "Custom (type RA and Dec)"

# Starting values (WDPN, Philadelphia ch 2, to a receiver near Washington DC; Geminids)
DEFAULTS = dict(
    tx_name="WDPN", tx_lat=40.04169, tx_lon=-75.23614, tx_h=378.9,
    rx_name="RX near Washington DC", rx_lat=38.821708, rx_lon=-77.103720, rx_h=100.0,
    freq=54.309441,
    shower="Geminids", radiant_name="Geminids", ra=112.0, dec=33.0,
    day=date(2026, 12, 14), hour=0, minute=0,
)
STATE = dict(DEFAULTS)     # what the form holds now; survives re-running the form cell

_DESC_W = {"description_width": "120px"}
_FIELD = w.Layout(width="290px")
_BOX = w.Layout(border="1px solid #8885", padding="6px 10px", margin="0 8px 8px 0")


def _check(v):
    """Raise ValueError with a plain message if the inputs cannot be used."""
    for k, x in v.items():
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            raise ValueError("A field is empty or not a number.")
    if not (-90 <= v["tx_lat"] <= 90 and -90 <= v["rx_lat"] <= 90):
        raise ValueError("Latitudes must be between -90 and 90 (north is positive).")
    if not (-180 <= v["tx_lon"] <= 180 and -180 <= v["rx_lon"] <= 180):
        raise ValueError("Longitudes must be between -180 and 180 (west is negative, so US longitudes are negative).")
    if v["freq"] <= 0:
        raise ValueError("The frequency must be positive.")
    if not 0 <= v["ra"] <= 360:
        raise ValueError("Right ascension must be between 0 and 360 degrees.")
    if not -90 <= v["dec"] <= 90:
        raise ValueError("Declination must be between -90 and 90 degrees.")
    d = mc.baseline_between((v["tx_lat"], v["tx_lon"]), (v["rx_lat"], v["rx_lon"]))
    if d < 5:
        raise ValueError(f"TX and RX are only {d:.1f} km apart; check the coordinates.")
    if d > 3000:
        raise ValueError(f"The baseline is {d:.0f} km; check the coordinates.")


def _when(v):
    return datetime(v["day"].year, v["day"].month, v["day"].day, int(v["hour"]), int(v["minute"]), tzinfo=timezone.utc)


def values():
    """The validated form values, for the cells after the form. Raises ValueError (with the reason) if a field is bad."""
    v = dict(STATE)
    if v["day"] is None:
        raise ValueError("Choose a date.")
    try:
        _check({k: x for k, x in v.items() if k not in ("tx_name", "rx_name", "shower", "radiant_name")})
    except ValueError as err:
        raise ValueError("Check the form in Step 1: " + str(err)) from None
    return SimpleNamespace(
        tx_name=v["tx_name"] or "TX", tx=(v["tx_lat"], v["tx_lon"], v["tx_h"]),
        rx_name=v["rx_name"] or "RX", rx=(v["rx_lat"], v["rx_lon"], v["rx_h"]),
        freq_mhz=v["freq"], radiant_name=v["radiant_name"] or "radiant", ra_deg=v["ra"], dec_deg=v["dec"], when=_when(v))


def show_form():
    """Build and display the form, filled with the values it holds now."""
    s = STATE
    txt = lambda label, key, tip: w.Text(value=s[key], description=label, tooltip=tip, style=_DESC_W, layout=_FIELD)
    num = lambda label, key, tip: w.FloatText(value=s[key], description=label, tooltip=tip, style=_DESC_W, layout=_FIELD)
    lat_tip = "Decimal degrees. North is positive."
    lon_tip = "Decimal degrees. East is positive, so US longitudes are negative."
    f = dict(
        tx_name=txt("Name", "tx_name", "A label for the transmitter (illuminator)."),
        tx_lat=num("Latitude (deg)", "tx_lat", lat_tip), tx_lon=num("Longitude (deg)", "tx_lon", lon_tip),
        tx_h=num("Height (m)", "tx_h", "Metres above sea level: ground elevation plus tower height."),
        rx_name=txt("Name", "rx_name", "A label for the receiver."),
        rx_lat=num("Latitude (deg)", "rx_lat", lat_tip), rx_lon=num("Longitude (deg)", "rx_lon", lon_tip),
        rx_h=num("Height (m)", "rx_h", "Metres above sea level: ground elevation plus antenna height."),
        freq=num("Frequency (MHz)", "freq", "The illuminator frequency."),
        radiant_name=txt("Radiant name", "radiant_name", "A label for the plots."),
        ra=num("RA (deg, J2000)", "ra", "Right ascension of the radiant in degrees, 0 to 360."),
        dec=num("Dec (deg, J2000)", "dec", "Declination of the radiant in degrees, -90 to 90."),
    )
    shower = w.Dropdown(options=[CUSTOM] + list(SHOWERS), value=s["shower"] if s["shower"] in SHOWERS else CUSTOM,
                        description="Shower", style=_DESC_W, layout=_FIELD,
                        tooltip="Fills in the name, RA and Dec with approximate radiant values. You can edit them afterwards.")
    day = w.DatePicker(value=s["day"], description="Date (UTC)", style=_DESC_W, layout=_FIELD,
                       tooltip="Steps 4, 5, 6b and 7 look at this whole UTC day. The other steps use the time below.")
    hour = w.BoundedIntText(value=s["hour"], min=0, max=23, description="Hour (UTC)", style=_DESC_W, layout=_FIELD)
    minute = w.BoundedIntText(value=s["minute"], min=0, max=59, description="Minute", style=_DESC_W, layout=_FIELD)
    reset = w.Button(description="Reset to the example", icon="undo", tooltip="Put back WDPN to a receiver near Washington DC, Geminids.")
    swap = w.Button(description="Swap TX / RX", icon="exchange", tooltip="Exchange the transmitter and receiver fields.")
    readout = w.HTML()
    state = {"dirty": False, "quiet": False}
    widgets = dict(f, day=day, hour=hour, minute=minute)

    def update_readout():
        try:
            v = dict(STATE)
            if v["day"] is None:
                raise ValueError("Choose a date.")
            _check({k: x for k, x in v.items() if k not in ("tx_name", "rx_name", "shower", "radiant_name")})
        except ValueError as err:
            readout.value = f"<span style='color:#b42318'>&#9888; {err}</span>"
            return
        tx, rx = (v["tx_lat"], v["tx_lon"]), (v["rx_lat"], v["rx_lon"])
        brg = mc.bearing_deg(tx, rx)
        mid = md.ecef_to_geodetic((mc.ecef(*tx, 0.0) + mc.ecef(*rx, 0.0)) / 2)
        when = _when(v)
        el = md.heading_descent(md.trail_direction(v["ra"], v["dec"], when), float(mid[0]), float(mid[1]))[1]
        text = (f"Baseline <b>{mc.baseline_between(tx, rx):.1f} km</b>, receiver at bearing {brg:.0f}&deg; ({mc.compass(brg)}) from the transmitter. "
                f"At {when:%Y-%m-%d %H:%M} UTC the radiant is ")
        text += (f"<b>{el:.0f}&deg; above</b> the horizon at the middle of the path." if el > 0 else
                 "<b>below the horizon</b> at the middle of the path (Steps 4, 5, 6b and 7 still scan the whole day).")
        if state["dirty"]:
            text += " <b>Fields changed: choose Run All to refresh the results below.</b>"
        readout.value = f"<span style='opacity:.85'>{text}</span>"

    def on_change(key):
        def handler(change):
            STATE[key] = change["new"]
            if not state["quiet"]:
                state["dirty"] = True
            if key in ("ra", "dec") and not state["quiet"]:      # typing a radiant by hand makes it a custom one
                if shower.value != CUSTOM and SHOWERS[shower.value] != (STATE["ra"], STATE["dec"]):
                    state["quiet"] = True
                    shower.value = CUSTOM
                    state["quiet"] = False
            update_readout()
        return handler

    for k, wid in widgets.items():
        wid.observe(on_change(k), names="value")

    def on_shower(change):
        STATE["shower"] = change["new"]
        if change["new"] in SHOWERS:
            ra, dec = SHOWERS[change["new"]]
            state["quiet"] = True
            f["radiant_name"].value, f["ra"].value, f["dec"].value = change["new"], ra, dec
            state["quiet"] = False
            STATE.update(radiant_name=change["new"], ra=ra, dec=dec)
        state["dirty"] = True
        update_readout()

    shower.observe(on_shower, names="value")

    def on_reset(_):
        state["quiet"] = True
        for k, wid in widgets.items():
            wid.value = DEFAULTS[k]
        shower.value = DEFAULTS["shower"]
        state["quiet"] = False
        STATE.update(DEFAULTS)
        state["dirty"] = True
        update_readout()

    def on_swap(_):
        state["quiet"] = True
        for a, b in (("tx_name", "rx_name"), ("tx_lat", "rx_lat"), ("tx_lon", "rx_lon"), ("tx_h", "rx_h")):
            f[a].value, f[b].value = f[b].value, f[a].value
            STATE[a], STATE[b] = f[a].value, f[b].value
        state["quiet"] = False
        state["dirty"] = True
        update_readout()

    reset.on_click(on_reset)
    swap.on_click(on_swap)
    update_readout()
    box = lambda title, *fields: w.VBox([w.HTML(f"<b>{title}</b>"), *fields], layout=_BOX)
    display(w.VBox([
        w.HBox([box("Transmitter (illuminator)", f["tx_name"], f["tx_lat"], f["tx_lon"], f["tx_h"], f["freq"]),
                box("Receiver", f["rx_name"], f["rx_lat"], f["rx_lon"], f["rx_h"]),
                box("Radiant and time", shower, f["radiant_name"], f["ra"], f["dec"], day, hour, minute)],
               layout=w.Layout(flex_flow="row wrap")),
        w.HBox([reset, swap]),
        readout,
    ]))
