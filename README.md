# Poseidon Mission Sandbox

Lightweight SSV/AUV mission playback sandbox built around YAML scenarios, a small surrogate dynamics stack, and a thin Streamlit review UI.

## What is in this repo

- `mission_analysis.py` — authoritative mission runner
- `guidance_core.py` — controllers, plant models, logging, and Matplotlib animation helpers
- `forcing_provider.py` — weather/current forcing from `noaa_44014_2012.json`
- `bathymetry_sensor_clean_slate.py` — optional bathymetry sensing and map-aided correction hooks
- `sim_config.yaml` — runtime defaults
- `scenario-mvp-*.yaml` — mission scenarios
- `streamlit_app.py` — operator-facing app for configuring runs and reviewing results

## What it does

- runs scenarios end to end for surface and underwater vehicles
- switches between `direct_pursuit` and `damped_pursuit`
- applies weather modes `min`, `minus_1sd`, and `minus_2sd`
- toggles AUV IMU drift and bathymetry sensing
- writes one CSV per vehicle under `logs/`
- visualizes planned vs actual vs estimated motion in 2D and 3D
- can export 2D and 3D playback GIFs from the Streamlit UI

## Run it

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

CLI runner:

```bash
python mission_analysis.py scenario-mvp-02-short-scan.yaml --no-plot --no-animate
```

## Notes on the current repo state

- The PNG maps in the repo are used as static context images in the UI.
- Weather forcing is wired to the local JSON dataset already included here.
- Bathymetry matching only becomes active when a matching GeoTIFF is present; without one, the rest of the mission runner still works.
