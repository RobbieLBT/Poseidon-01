# Poseidon-01

Poseidon-01 is a lightweight mission analysis and autonomy-evaluation environment for autonomous surface and underwater vehicles. It is built around a simple question: **what happens at the mission level when vehicle physics, guidance laws, sensing, estimation, and environment all start interacting?**

The project started as a mission playback and guidance-validation effort. It has since grown into a compact sandbox for comparing route logic, controller behavior, environmental forcing, and navigation-stack choices against the same mission definition. The current stack is intentionally lightweight and surrogate-based, but it is already useful for exposing the kinds of system-level behaviors that are easy to miss when each subsystem is tuned in isolation.

## What this repo is for

Poseidon-01 is meant for early-stage analysis, architecture validation, and technical trade studies of software-defined autonomous martime vehicles.

In its current form, the repo lets you:

- define missions in YAML,
- simulate mixed SSV/AUV missions with event-driven sequencing,
- switch between lightweight guidance strategies,
- apply simple surface and underwater current forcing,
- inject navigation drift,
- test DVL and bathymetry-aided localization hooks,
- log per-vehicle state histories and mission events, and
- review results in either a CLI workflow or a Streamlit dashboard.

This is **not** a high-fidelity hydrodynamics environment, a production autonomy stack, or a completed digital twin. It is a conceptual-stage analysis tool designed to make interactions visible early, while the architecture is still flexible.

## Current capabilities

### Mission definition and playback

- Scenario-driven mission execution from YAML.
- Multi-vehicle missions with attachments, detach/reattach actions, event triggers, waypoint-triggered progression, and loiter segments.
- Point and polyline route geometry.

### Guidance and plant modeling

- Two pursuit-style guidance options: `direct_pursuit` and `damped_pursuit`.
- Lightweight surrogate plant models for:
  - a surface vehicle with heading/rudder/thrust behavior, and
  - an underwater vehicle with added depth/elevator/buoyancy channels.
- Waypoint capture logic with configurable acceptance radii, reacquire radii, and slowdown radius behavior.

### Environment and forcing

- Scenario-level environmental forcing generated from NOAA buoy data.
- Separate surface and underwater current vectors.
- A time-aware forcing-provider interface, even though the current JSON-backed provider is still static within a scenario block.

### Navigation and sensing hooks

- IMU-like drift propagation in the state estimate.
- DVL bottom-lock correction logic.
- Bathymetry-aided localization using a cone-style seafloor sensing surrogate and map matching against a GeoTIFF.
- Clean separation between sensing, estimation correction, control, and plant propagation.

### Outputs and review

- One CSV log per vehicle for downstream analysis.
- 2D and 3D mission plots and playback animations.
- Streamlit review UI with:
  - mission preview,
  - run configuration controls,
  - mission KPIs,
  - state/error/control time histories,
  - DVL and bathymetry modifier plots,
  - optional GIF export, and
  - a simple wall-clock compute-energy proxy.

## What the repo does not claim yet

A few things are important to say directly:

- The vehicle models are notional surrogate plants, not calibrated high-fidelity dynamics.
- The weather/current model is presently scenario-based, not a full time-varying atmospheric or ocean field.
- Bathymetry localization is a map-matching surrogate, not a full estimator implementation.
- The log schema anticipates more sensor channels than the current mission loop fully exploits. In the present code, the visible localization hooks are centered on drift propagation, DVL, and bathymetry.

That scope is deliberate. The value here is in exposing interaction and architecture early, not pretending to be a finished end-state simulator.

## Logical repo structure

The runtime code expects a package-style layout roughly like this:

```text
Poseidon-01/
├─ mission_analysis.py
├─ streamlit_app.py
├─ sim_config.yaml
├─ requirements.txt
├─ guidance_and_control/
│  ├─ guidance_core.py
│  └─ sensors/
│     └─ bathymetry_sensor_clean_slate.py
├─ environment/
│  ├─ forcing_provider.py
│  ├─ maps/
│  │  ├─ Pamlico_Sound.png
│  │  ├─ Hampton_Roads.png
│  │  ├─ Hudson_Canyon.png
│  │  └─ matching GeoTIFFs when bathymetry localization is enabled
│  └─ weather/
│     ├─ noaa_44014_2012.csv
│     └─ noaa_44014_2012.json
├─ scenarios/
│  └─ mission_profiles/
│     └─ scenario-mvp-*.yaml
└─ logs/
   └─ simulator_run/
```

A few utility scripts also sit alongside the core runtime:

- `dynamic_weather_model.py` builds scenario forcing blocks from a NOAA buoy CSV.
- `read_resolution_updated.py` is a bathymetry GeoTIFF sanity-check and texture-characterization utility.
- `read_resolution.py` is an earlier version of the same idea.

## Core mechanics

The easiest way to understand Poseidon-01 is to follow one mission run from input to output.

### 1. Scenario YAML defines mission geometry and sequencing

Each scenario defines vehicles, initial conditions, route geometry, and event logic.

At a high level, each route item can contain:

- an `id`,
- a `trigger` such as `immediate`, `after_waypoint`, or `event`,
- either a single `position` or a polyline `geometry`,
- a `mode` such as `continue` or `loiter`,
- route parameters like `leg_speed_mps` or `loiter_seconds`, and
- optional `on_trigger` / `on_complete` actions such as `detach`, `attach_to`, or `emit_event`.

A minimal pattern looks like this:

```yaml
vehicles:
  auv_1:
    initial_state:
      pose:
        frame: local_ned
        value: [0, 0, 0]
    route:
      - id: deploy
        trigger: { type: immediate }
        position: { frame: local_ned, value: [100, 25, -20] }
        mode: loiter
        params: { loiter_seconds: 5 }
      - id: survey_run
        trigger: { type: after_waypoint, waypoint_id: deploy }
        geometry:
          type: polyline
          frame: local_ned
          points:
            - [200, 50, -20]
            - [200, 200, -20]
```

This structure is what makes the repo useful for mission-level review instead of only single-leg trajectory playback.

### 2. `sim_config.yaml` defines runtime behavior

The simulation config carries the runtime knobs that you want to change independently from mission geometry:

- integration, control, and logging rates,
- per-vehicle simulation profiles,
- vehicle-mode selection (`surface` vs `underwater`),
- controller choice,
- plant coefficients and gains,
- drift settings,
- sensor settings, and
- estimation/blending settings.

This separation matters. The same mission geometry can be re-run under different control laws, current scenarios, or sensor suites without rewriting the scenario itself.

### 3. The mission runner builds vehicle states and executes the scenario

`mission_analysis.py` is the authoritative mission runner.

For each vehicle it:

- builds a runtime config from `sim_config.yaml`,
- initializes truth and estimate state,
- parses the route into internal route items,
- evaluates triggers and actions,
- handles attachment semantics,
- advances waypoint-by-waypoint through the active route item, and
- logs mission state as the run progresses.

This is the layer that turns a static mission definition into a stateful, event-driven run.

### 4. Guidance and plant propagation happen in `guidance_core.py`

The guidance core owns the low-level mechanics that sit beneath the mission runner:

- command generation,
- surface and underwater surrogate dynamics,
- estimate propagation with drift,
- waypoint acceptance logic,
- CSV writing, and
- Matplotlib plotting/animation helpers.

The important architectural choice here is separation. The mission runner decides **what** the vehicle should do next; the guidance core decides **how** the vehicle moves in response.

### 5. Environmental forcing is layered in through a provider

The current provider (`environment/forcing_provider.py`) reads a JSON payload built from NOAA buoy statistics and returns the applicable current vector for the vehicle mode.

Right now this is lightweight by design:

- forcing is scenario-based,
- the provider interface is time-aware, but
- the shipped JSON provider currently returns static forcing for the selected scenario block.

That still makes the environment operationally relevant without forcing the repo into premature complexity.

### 6. Navigation state is allowed to drift, then corrected

The estimate state is propagated separately from truth. That is where the repo becomes more than a playback tool.

Current navigation behavior is organized around three ideas:

- **drift propagation** introduces navigation error into the estimate,
- **DVL logic** tightens the estimate using bottom-lock velocity information, and
- **bathymetry localization** attempts a texture-based correction against a preloaded seafloor map.

Bathymetry correction only becomes active when all of the following are true:

- the bathymetry sensor is enabled,
- bathymetry localization is enabled in estimation settings, and
- a matching GeoTIFF is available.

That correction path is intentionally modular. The sensing model, the matching logic, the correction gain, and the plant/controller all remain distinct.

### 7. Structured logs drive downstream review

Each vehicle writes a CSV with truth state, estimated state, commands, waypoint distance, navigation debug values, DVL/bathymetry fields, and event rows.

That log-first design is important. Poseidon-01 is not just a visualization toy. It is built so that runs can be compared, filtered, post-processed, and turned into KPIs.

## Getting started

### Install

```bash
pip install -r requirements.txt
```

### Run the Streamlit dashboard

```bash
streamlit run streamlit_app.py
```

The UI lets you select:

- mission scenario,
- static map PNG,
- weather mode,
- control strategy,
- IMU drift on/off,
- DVL on/off,
- bathymetry on/off,
- bathymetry blend gain,
- playback speed,
- max mission time, and
- optional GIF export.

### Run from the CLI

```bash
python mission_analysis.py scenarios/mission_profiles/scenario-mvp-02-short-scan.yaml --no-plot --no-animate
```

Useful flags:

```bash
--sim-config path/to/sim_config.yaml
--dt 0.1
--max-time 1200
--output-dir logs/simulator_run/test_case
--no-plot
--no-animate
--playback-speed 5.0
```

## Weather and environment workflow

The repo includes a simple utility for turning NOAA buoy data into scenario blocks:

```bash
python dynamic_weather_model.py environment/weather/noaa_44014_2012.csv \
  --current-units cm/s \
  --json-out environment/weather/noaa_44014_2012.json
```

That JSON payload is then referenced by `sim_config.yaml` under `environment_forcing`.

This is a lightweight but useful pattern:

1. pull representative real-world data,
2. compress it into a few scenario blocks,
3. keep mission definitions fixed, and
4. compare autonomy/control behavior across environmental cases.

## Bathymetry workflow

Bathymetry support is split into two pieces:

1. **visual context** via PNG maps in the UI, and
2. **localization support** via GeoTIFF bathymetry rasters for the sensing/matching logic.

If you want bathymetry localization to actually run:

- place the GeoTIFF beside the selected PNG with the same stem,
- enable the bathymetry sensor,
- enable bathymetry localization in the estimation config, and
- point the sensor config at the GeoTIFF path.

A useful sanity-check step is:

```bash
python read_resolution_updated.py environment/maps/Pamlico_Sound.tiff --debug
```

That utility helps confirm raster resolution, map extent, and whether the local seafloor texture is likely to produce a useful bathymetric correction.

## How to review the repo quickly

For a technical reviewer, the fastest path is:

1. Read `mission_analysis.py` to understand the execution loop and route/event semantics.
2. Read `guidance_core.py` to understand the controller, plant, estimate, and logging interfaces.
3. Read `sim_config.yaml` to see the actual tuning knobs.
4. Open one scenario YAML and trace the mission logic.
5. Run the Streamlit app and compare the same scenario under different control/sensor/weather settings.
6. Inspect the CSV logs and KPI plots rather than relying only on the animations.

That review order usually makes the architecture clear much faster than starting from the UI alone.

## Why this repo is useful

Poseidon-01 is most valuable when you care about **interaction effects**:

- guidance laws that look reasonable locally but miss waypoints at mission scale,
- control limits that create emergent orbiting or sluggish capture,
- navigation improvements that depend strongly on environment,
- sensors that do not simply add linearly when blended together, and
- compute cost that becomes part of the design problem rather than an afterthought.

That is the level at which software-defined autonomous systems start to get interesting.

## Current caveats and likely next steps

The current stack already supports useful analysis, but a few next steps are obvious:

- richer sensor models,
- more mature navigation estimation,
- tighter path-management logic,
- in-simulator vehicle physics upgrades,
- more realistic dynamic weather/ocean forcing,
- larger-scale parametric run automation, and
- longer-horizon digital-twin style lifecycle studies.

## Bottom line

Poseidon-01 is a compact environment for testing mission logic, guidance behavior, sensor interaction, and environmental sensitivity in one place. Its main value is not raw fidelity. Its main value is that it helps surface system-level behavior early enough for that behavior to still change the design.
