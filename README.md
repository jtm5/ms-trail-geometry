# Meteor trail geometry: can a radiant give a specular echo?

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jtm5/ms-trail-geometry/HEAD?labpath=meteor_detect_demo.ipynb)

A small notebook for meteor-scatter (MS) planning. For a transmitter (illuminator), a receiver, a meteor shower radiant and a time, it asks where meteor trails from that radiant can give a **specular echo**, and shows how that changes through the day.

**Early prototype, not validated.** It has not yet been checked against logged echoes in any way that supports forecasting. A first comparison with one Perseid log was weak for the optional strength steps. Feedback from people who run MS stations is the point of sharing it.

## Try it

Click the Binder badge above (it may take a minute or two to start), then open the notebook and run it. No installation is needed.

To run it on your own machine: `pip install -r requirements.txt`, then open `meteor_detect_demo.ipynb` in Jupyter or VS Code.

## Use your own stations and radiant

**Step 1** of the notebook is a form. Fill it in and choose **Run All**; the form keeps what you typed when you run again.

- **Transmitter (illuminator)** and **Receiver**: a name, latitude and longitude in degrees (north and east are positive, so US longitudes are negative) and height above sea level in metres. The transmitter box also holds the frequency in MHz. **Swap TX / RX** exchanges the two.
- **Radiant and time**: pick a shower from the menu to fill in an approximate radiant, or type your own J2000 right ascension and declination in degrees. Then set the date and time in UTC. Steps 4, 5, 6b and 7 look at the whole UTC day of the date; the other steps use the time you give. Menu radiants are approximate values near peak. A radiant drifts from night to night, so check a current list such as the IMO meteor shower calendar for your date.
- Under the form, a cell holds the other settings as named values (height band, direction tolerance, plot sizes, link margins), each with a comment.
- **Reset to the example** puts back WDPN to a receiver near Washington DC, Geminids.

A line under the form shows the baseline length, the bearing, and whether the radiant is above the horizon at the time you chose.

## What it does

- **Step 2, one trail.** Tests a single trail for a specular point in the ionised height band (default 70 to 120 km) above both horizons.
- **Step 3, all trails from the radiant, on a map.** Green: a specular point exists. Amber: only if the trail direction is off by up to a tolerance. Red: none.
- **Step 4, through the day.** The share of trails that pass, for every half hour.
- **Step 5, relative rate index (optional).** Pass area times the sine of the radiant elevation, a rough guide to how many passing trails arrive per hour.
- **Step 6, relative strength (optional, prototype).** A relative link margin in dB from the bistatic range equation, as a map, a zoomed map and a profile against distance from the middle of the baseline. **Step 6b** is an hour-by-hour set of the zoomed maps on one colour scale.
- **Step 6c, polarization and trail aspect (optional, prototype).** Adds two more geometry terms, still with isotropic antennas, assuming horizontal polarization at both ends and no Faraday rotation, so you can see where polarization nulls fall.
- **Step 7, rate index within a link margin (optional).** Counts only trails within a margin you choose of the day's strongest trail. The notebook does not suggest a margin. Pick what suits your station.

## What it does not do

It is geometry only. It does not model antenna patterns, gain, the noise floor, how many meteors are bright enough, the radiant's spread, zenith attraction, Earth's rotation, meteor deceleration, or how long a trail lasts. Strength values are **relative** (dB relative to the strongest trail), never absolute. Read the notes under each step before drawing conclusions.

## Files

- `meteor_detect_demo.ipynb`: the notebook.
- `meteor_detect.py`: the detection and strength code the notebook uses.
- `meteor_polar.py`: the optional polarization and aspect prototype (Step 6c). Delete it with that step if you don't want it.
- `meteor_form.py`: the input form in Step 1.
- `meteor_core.py`: shared geometry helpers.

## Maps

Step 3 draws map tiles from Esri's World Topo Map service ("Map tiles © Esri"). The first run downloads them (internet required) and keeps them in `.tile_cache/`. Set `MAP_TILES = None` in `meteor_core.py` for a plain background.

## Feedback

Please open an issue, or tell me what you tried and what looked wrong or missing. Useful: stations and showers where you know what you would expect to see, and which of the optional steps you would keep or drop.

## Licence

MIT. See `LICENSE`.
