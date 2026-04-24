"""Animate and validate scenario YAML mission geometry and event sequencing.

Supports two playback modes per vehicle:
- kinematic: straight-line rectilinear playback
- dynamic: controller + plant response via guidance_core.py

Mission geometry, triggers, events, loitering, and attachment semantics remain driven by
scenario YAML. Runtime behavior is configured through sim_config.yaml.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

DISPLAY_OFFSET_M = 1.5
DEFAULT_TIME_STEP_S = 1.0
DEFAULT_MAX_SIM_TIME_S = 25000.0
DEFAULT_LOITER_SECONDS = 15.0
POSITION_EPSILON_M = 0.5

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required. Install it with: pip install pyyaml") from exc

from guidance_and_control.guidance_core import (
    Command,
    EnvForcing,
    EstimateState,
    TruthState,
    animate_mission_2d,
    animate_mission_3d,
    append_estimate_history,
    append_truth_history,
    compute_guidance_command,
    get_environment_forcing,
    get_global_setting,
    get_vehicle_runtime_config,
    init_estimate_history,
    init_history,
    initial_truth_from_pose,
    load_sim_config,
    make_event_log_row,
    make_vehicle_log_row,
    plot_mission_2d,
    plot_mission_3d,
    propagate_estimate,
    step_vehicle_plant,
    truth_to_estimate,
    waypoint_distance,
    waypoint_reached,
    wrap_angle,
    write_vehicle_csv,
    apply_bathy_drift_correction
)
from guidance_and_control.sensors.bathymetry_sensor_clean_slate import (
    BathymetryGrid,
    ConeVolumeConfig,
    BathyCorrectionConfig,
    bathy_update_step,
)

from environment.forcing_provider import JsonScenarioForcingProvider

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SIM_CONFIG_PATH = REPO_ROOT / 'sim_config.yaml'
DEFAULT_LOG_DIR = REPO_ROOT / 'logs' / 'simulator_run'


@dataclass
class RouteItem:
    route_id: str
    trigger: Dict[str, Any]
    mode: str
    points: List[Tuple[float, float, float]]
    geometry_type: str = "point"
    params: Dict[str, Any] = field(default_factory=dict)
    emit_event: Optional[str] = None
    on_trigger: List[Dict[str, Any]] = field(default_factory=list)
    on_complete: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class VehicleState:
    name: str
    route: List[RouteItem]
    pos: List[float]
    vel: List[float]
    attached: bool = False
    parent_vehicle: Optional[str] = None
    attachment_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    current_index: Optional[int] = None
    current_trigger_executed: bool = False
    current_point_index: int = 0
    completed_waypoints: Set[str] = field(default_factory=set)
    completed_once: Set[str] = field(default_factory=set)
    waiting_since: Optional[float] = None
    done: bool = False
    runtime_cfg: Dict[str, Any] = field(default_factory=dict)
    truth: TruthState = field(default_factory=TruthState)
    estimate: EstimateState = field(default_factory=EstimateState)
    command: Command = field(default_factory=Command)
    truth_history: Dict[str, List[float]] = field(default_factory=init_history)
    estimate_history: Dict[str, List[float]] = field(default_factory=init_estimate_history)
    log_rows: List[Dict[str, Any]] = field(default_factory=list)
    rng: Any = None
    next_control_time_s: float = 0.0
    next_log_time_s: float = 0.0
    next_sensor_time_s: float = 0.0
    next_dvl_time_s: float = 0.0
    next_bathy_time_s: float = 0.0
    current_target_xyz: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    nav_stack: Any = None
    nav_debug: Dict[str, Any] = field(default_factory=dict)
    bathy_debug: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "status": "disabled", "update_applied": 0})
    last_bathy_measurement: Optional[Dict[str, Any]] = None


class ScenarioAnimator:
    def __init__(
        self,
        scenario: Dict[str, Any],
        sim_cfg: Dict[str, Any],
        time_step_s: float,
        output_dir: Optional[Path] = None,
    ):
        self.scenario = scenario
        self.sim_cfg = sim_cfg
        self.time_step_s = float(time_step_s)
        self.time_s = 0.0
        self.events: List[Tuple[float, str]] = []
        self.event_names: Set[str] = set()
        self.vehicle_states: Dict[str, VehicleState] = {}
        self.status_history: List[str] = []
        self.planned_paths: Dict[str, List[Tuple[float, float, float]]] = {}
        log_dir_value = sim_cfg.get("logging", {}).get("output_dir", str(DEFAULT_LOG_DIR))
        self.output_dir = output_dir or Path(log_dir_value)
        self.log_interval_s = float(sim_cfg.get("logging", {}).get("sample_interval_s", 1.0))
        self.bathy_grid_cache: Dict[str, BathymetryGrid] = {}
        self.forcing_provider = JsonScenarioForcingProvider(
            self.sim_cfg,
            sim_cfg_source_path=self.sim_cfg.get("__source_path__"),
        )
        self._build_vehicle_states()
        self._sync_attached_vehicles()
        self._snapshot()

    def _get_env_forcing(self, runtime_cfg: Dict[str, Any], t_s: float) -> EnvForcing:
        vehicle_mode = str(runtime_cfg.get("vehicle_mode", "surface")).lower()

        if getattr(self.forcing_provider, "mode", "none") == "json_scenario":
            return self.forcing_provider.forcing_at_time(t_s, vehicle_mode)

        return get_environment_forcing(
            runtime_cfg.get("environment", {}),
            vehicle_mode,
        )

    def _build_vehicle_states(self) -> None:
        vehicles = self.scenario.get("vehicles", {})
        for vehicle_name, vehicle_cfg in vehicles.items():
            route = [self._parse_route_item(item) for item in vehicle_cfg.get("route", [])]
            initial_state = vehicle_cfg.get("initial_state", {})
            pos, attached, parent_vehicle, offset = self._parse_initial_pose(initial_state, route)
            vel = self._parse_initial_velocity(initial_state)
            runtime_cfg = get_vehicle_runtime_config(self.sim_cfg, vehicle_name)

            print(f"[{vehicle_name}] sim config source: {self.sim_cfg.get('__source_path__')}")
            print(f"[{vehicle_name}] bathy cfg: {runtime_cfg.get('sensors', {}).get('bathymetry_cone')}")
            print(f"[{vehicle_name}] estimation cfg: {runtime_cfg.get('estimation', {})}")

            truth = initial_truth_from_pose(pos)
            estimate = truth_to_estimate(truth)
            rng = np.random.default_rng(int(runtime_cfg.get("drift", {}).get("seed", 7)))
            nav_stack = None

            nav_stack = None
            estimate = truth_to_estimate(truth)


            state = VehicleState(
                name=vehicle_name,
                route=route,
                pos=list(pos),
                vel=list(vel),
                attached=attached,
                parent_vehicle=parent_vehicle,
                attachment_offset=offset,
                runtime_cfg=runtime_cfg,
                truth=truth,
                estimate=estimate,
                rng=rng,
                next_control_time_s=0.0,
                next_log_time_s=0.0,
                next_sensor_time_s=0.0,
                next_dvl_time_s=0.0,
                next_bathy_time_s=0.0,
                nav_stack=nav_stack,
            )
            self.vehicle_states[vehicle_name] = state
            self.planned_paths[vehicle_name] = self._extract_planned_path(route)

            print(f"{vehicle_name} runtime_cfg controller:", runtime_cfg.get("controller"))
            print(f"{vehicle_name} coeffs:", runtime_cfg.get("coeffs"))
            print(f"{vehicle_name} gains:", runtime_cfg.get("gains"))
            print(f"{vehicle_name} depth enabled:", runtime_cfg.get("depth_channel_enabled"))
            print(f"{vehicle_name} initial target path count:", len(self.planned_paths[vehicle_name]))

    def _extract_planned_path(self, route: List[RouteItem]) -> List[Tuple[float, float, float]]:
        pts: List[Tuple[float, float, float]] = []
        for item in route:
            pts.extend(item.points)
        return pts

    def _parse_route_item(self, item: Dict[str, Any]) -> RouteItem:
        route_id = str(item.get("id") or item.get("route_id") or "route")
        trigger = item.get("trigger", {"type": "immediate"})
        mode = str(item.get("mode", "goto"))
        emit_event = item.get("emit_event")
        on_trigger = item.get("on_trigger", [])
        on_complete = item.get("on_complete", [])
        params = item.get("params", {})

        if "geometry" in item:
            geometry = item["geometry"]
            geometry_type = str(geometry.get("type", "point"))
            if geometry_type == "polyline":
                points = [tuple(map(float, p[:3])) for p in geometry.get("points", [])]
            else:
                p = geometry.get("value", [0.0, 0.0, 0.0])
                points = [tuple(map(float, p[:3]))]
        elif "position" in item:
            geometry_type = "point"
            p = item["position"].get("value", [0.0, 0.0, 0.0])
            points = [tuple(map(float, p[:3]))]
        else:
            geometry_type = "point"
            points = [(0.0, 0.0, 0.0)]

        return RouteItem(
            route_id=route_id,
            trigger=trigger,
            mode=mode,
            points=points,
            geometry_type=geometry_type,
            params=params,
            emit_event=emit_event,
            on_trigger=on_trigger,
            on_complete=on_complete,
        )

    def _parse_initial_pose(
        self,
        initial_state: Dict[str, Any],
        route: List[RouteItem],
    ) -> Tuple[Tuple[float, float, float], bool, Optional[str], Tuple[float, float, float]]:
        attached = False
        parent_vehicle = None
        offset = (0.0, 0.0, 0.0)

        attach = initial_state.get("attached_to")
        if attach:
            attached = True
            parent_vehicle = str(attach.get("vehicle"))
            offset_value = attach.get("offset", [0.0, 0.0, 0.0])
            offset = tuple(map(float, offset_value[:3]))
            return (0.0, 0.0, 0.0), attached, parent_vehicle, offset

        pose = initial_state.get("pose", {})
        if pose.get("frame") == "local_ned":
            value = pose.get("value", [0.0, 0.0, 0.0])
            return tuple(map(float, value[:3])), attached, parent_vehicle, offset

        if route and route[0].points:
            return route[0].points[0], attached, parent_vehicle, offset

        return (0.0, 0.0, 0.0), attached, parent_vehicle, offset

    def _parse_initial_velocity(self, initial_state: Dict[str, Any]) -> Tuple[float, float, float]:
        velocity = initial_state.get("velocity", {})
        if velocity.get("frame") == "local_ned":
            value = velocity.get("value", [0.0, 0.0, 0.0])
            return tuple(map(float, value[:3]))
        return (0.0, 0.0, 0.0)

    def _sync_attached_vehicles(self) -> None:
        for state in self.vehicle_states.values():
            if state.attached and state.parent_vehicle in self.vehicle_states:
                parent = self.vehicle_states[state.parent_vehicle]
                state.pos = [
                    parent.pos[0] + state.attachment_offset[0],
                    parent.pos[1] + state.attachment_offset[1],
                    parent.pos[2] + state.attachment_offset[2],
                ]
                state.truth.x = state.pos[0]
                state.truth.y = state.pos[1]
                state.truth.z = state.pos[2]
                state.estimate.x = state.pos[0]
                state.estimate.y = state.pos[1]
                state.estimate.z = state.pos[2]

    def _snapshot(self) -> None:
        for state in self.vehicle_states.values():
            append_truth_history(state.truth_history, state.truth)
            append_estimate_history(state.estimate_history, state.estimate)
            
    def _emit_event(self, event_name: str) -> None:
        if not event_name:
            return
        self.events.append((self.time_s, event_name))
        self.event_names.add(event_name)
        self.status_history.append(f"[{self.time_s:8.1f}s] EVENT: {event_name}")

    def _trigger_condition_met(self, trigger: Dict[str, Any], state: VehicleState) -> bool:
        trigger_type = str(trigger.get("type", "immediate")).lower()

        if trigger_type == "immediate":
            return True

        if trigger_type == "time_gte":
            return self.time_s >= float(trigger.get("time_s", 0.0))

        if trigger_type == "event":
            event_name = str(trigger.get("name") or trigger.get("event_id") or "")
            return event_name in self.event_names

        if trigger_type == "waypoint_completed":
            return str(trigger.get("name", "")) in state.completed_waypoints

        if trigger_type == "after_waypoint":
            waypoint_id = str(trigger.get("waypoint_id", ""))
            return waypoint_id in state.completed_waypoints

        return False

    def _apply_action(self, state: VehicleState, action: Dict[str, Any]) -> None:
        action_type = str(action.get("type", ""))
        if action_type == "emit_event":
            self._emit_event(str(action.get("name", "")))
        elif action_type == "detach":
            state.attached = False
            state.parent_vehicle = None
        elif action_type == "attach_to":
            state.attached = True
            state.parent_vehicle = str(action.get("vehicle"))
            offset = action.get("offset", [0.0, 0.0, 0.0])
            state.attachment_offset = tuple(map(float, offset[:3]))
            self._sync_attached_vehicles()

    def _apply_actions(self, state: VehicleState, actions: List[Dict[str, Any]]) -> None:
        for action in actions:
            self._apply_action(state, action)

    def _resolve_bathy_grid_source(self, geotiff_path: str) -> str:
        raw = str(geotiff_path).strip()
        if raw.startswith(("http://", "https://")):
            return raw
        path = Path(raw).expanduser()
        if path.is_absolute():
            return str(path)
        return str((REPO_ROOT / path).resolve())

    def _get_bathy_grid(self, geotiff_path: Optional[str]) -> Optional[BathymetryGrid]:
        if not geotiff_path:
            return None
        source = self._resolve_bathy_grid_source(str(geotiff_path))
        key = source
        if key in self.bathy_grid_cache:
            return self.bathy_grid_cache[key]
        try:
            grid = BathymetryGrid.from_geotiff(source)
            self.bathy_grid_cache[key] = grid
            return grid
        except Exception as exc:
            print(f"[bathymetry] failed to load grid '{source}': {exc}")
            return None

    def _apply_dvl_localization(self, state: VehicleState, dt: float) -> None:
        sensors_cfg = state.runtime_cfg.get("sensors", {})
        dvl_cfg = sensors_cfg.get("dvl", {})

        if not bool(dvl_cfg.get("enabled", False)):
            state.nav_debug.update({
                "dvl_used": False,
                "dvl_accepted": False,
                "dvl_bottom_lock": False,
                "dvl_altitude_m": "",
                "dvl_innovation_norm_mps": "",
            })
            return

        sigma = dvl_cfg.get("sigma_mps", [0.05, 0.05, 0.06])
        if not isinstance(sigma, (list, tuple)) or len(sigma) < 3:
            sigma = [0.05, 0.05, 0.06]
        bias = dvl_cfg.get("bias_mps", [0.0, 0.0, 0.0])
        if not isinstance(bias, (list, tuple)) or len(bias) < 3:
            bias = [0.0, 0.0, 0.0]

        dropout_prob = float(dvl_cfg.get("dropout_prob", 0.0))
        min_altitude_m = float(dvl_cfg.get("min_altitude_m", 0.5))
        max_altitude_m = float(dvl_cfg.get("max_altitude_m", 1.0e9))
        assume_lock_without_map = bool(dvl_cfg.get("assume_lock_without_map", True))
        vel_blend_gain = float(state.runtime_cfg.get("estimation", {}).get("dvl_blend_gain", 0.35))
        pos_blend_gain = float(state.runtime_cfg.get("estimation", {}).get("dvl_position_blend_gain", 0.15))

        # When bathymetry is enabled, let DVL primarily stabilize velocity rather than
        # re-pulling position against a fresh bathy correction. This avoids the common
        # fight where bottom-lock velocity updates amplify a poor bathy position update.
        bathy_enabled = bool(sensors_cfg.get("bathymetry_cone", {}).get("enabled", False))
        bathy_recent_accept = bool(state.bathy_debug.get("update_applied", 0))
        bathy_conf = float(state.bathy_debug.get("confidence", 0.0) or 0.0)
        if bathy_enabled:
            pos_blend_gain *= 0.25
            if bathy_recent_accept or bathy_conf >= 0.25:
                pos_blend_gain = 0.0
                vel_blend_gain *= 0.75

        grid = None
        altitude_m = float('nan')
        geotiff_path = sensors_cfg.get("bathymetry_cone", {}).get("geotiff_path")
        if geotiff_path:
            grid = self._get_bathy_grid(geotiff_path)
        if grid is not None:
            bottom_z = grid.sample_elevation(float(state.truth.x), float(state.truth.y))
            if np.isfinite(bottom_z):
                altitude_m = float(state.truth.z - bottom_z)

        bottom_lock = bool(assume_lock_without_map) if not np.isfinite(altitude_m) else bool(min_altitude_m <= altitude_m <= max_altitude_m)
        used = bottom_lock and (float(state.rng.random()) >= dropout_prob)

        if not used:
            state.nav_debug.update({
                "dvl_used": True,
                "dvl_accepted": False,
                "dvl_bottom_lock": bottom_lock,
                "dvl_altitude_m": altitude_m if np.isfinite(altitude_m) else "",
                "dvl_innovation_norm_mps": "",
            })
            return

        true_vx = float(state.truth.u * np.cos(state.truth.psi) * np.cos(state.truth.theta))
        true_vy = float(state.truth.u * np.sin(state.truth.psi) * np.cos(state.truth.theta))
        true_vz = float(state.truth.w)
        meas_v = np.array([
            true_vx + float(bias[0]) + float(state.rng.normal(0.0, float(sigma[0]))),
            true_vy + float(bias[1]) + float(state.rng.normal(0.0, float(sigma[1]))),
            true_vz + float(bias[2]) + float(state.rng.normal(0.0, float(sigma[2]))),
        ], dtype=float)
        est_v = np.array([float(state.estimate.vx), float(state.estimate.vy), float(state.estimate.vz)], dtype=float)
        innovation = meas_v - est_v

        state.estimate.vx = float(state.estimate.vx + vel_blend_gain * innovation[0])
        state.estimate.vy = float(state.estimate.vy + vel_blend_gain * innovation[1])
        state.estimate.vz = float(state.estimate.vz + vel_blend_gain * innovation[2])

        state.estimate.drift_err_x = float(state.estimate.drift_err_x - pos_blend_gain * innovation[0] * dt)
        state.estimate.drift_err_y = float(state.estimate.drift_err_y - pos_blend_gain * innovation[1] * dt)
        state.estimate.drift_err_z = float(state.estimate.drift_err_z - pos_blend_gain * innovation[2] * dt)
        state.estimate.x = float(state.truth.x + state.estimate.drift_err_x)
        state.estimate.y = float(state.truth.y + state.estimate.drift_err_y)
        state.estimate.z = float(state.truth.z + state.estimate.drift_err_z)

        state.nav_debug.update({
            "dvl_used": True,
            "dvl_accepted": True,
            "dvl_bottom_lock": bottom_lock,
            "dvl_altitude_m": altitude_m if np.isfinite(altitude_m) else "",
            "dvl_innovation_norm_mps": float(np.linalg.norm(innovation)),
            "dvl_vel_blend_gain": vel_blend_gain,
            "dvl_pos_blend_gain": pos_blend_gain,
            "est_vx": float(state.estimate.vx),
            "est_vy": float(state.estimate.vy),
            "est_vz": float(state.estimate.vz),
        })

    def _apply_bathymetry_localization(self, state: VehicleState, dt: float) -> None:
        bathy_cfg = state.runtime_cfg.get("sensors", {}).get("bathymetry_cone", {})
        est_cfg = state.runtime_cfg.get("estimation", {})

        if not bathy_cfg.get("enabled", False):
            state.bathy_debug = {"enabled": False, "status": "disabled", "update_applied": 0}
            print(f"[{state.name}] BATHY OFF t={self.time_s:.1f} status=disabled")
            return

        geotiff_path = bathy_cfg.get("geotiff_path")
        grid = self._get_bathy_grid(geotiff_path)
        if grid is None:
            state.bathy_debug = {"enabled": True, "status": "grid_unavailable", "update_applied": 0}
            print(f"[{state.name}] BATHY OFF t={self.time_s:.1f} status=grid_unavailable path={geotiff_path}")
            return

        if not bool(est_cfg.get("bathymetry_localization_enabled", False)):
            state.bathy_debug = {"enabled": True, "status": "disabled_in_estimation", "update_applied": 0}
            print(f"[{state.name}] BATHY OFF t={self.time_s:.1f} status=disabled_in_estimation")
            return

        sensor_cfg = ConeVolumeConfig(
            enabled=True,
            num_beams_azimuth=int(bathy_cfg.get("num_beams_azimuth", 16)),
            num_beams_radial=int(bathy_cfg.get("num_beams_radial", 4)),
            max_slant_range_m=float(bathy_cfg.get("max_slant_range_m", 100.0)),
            min_slant_range_m=float(bathy_cfg.get("min_slant_range_m", 1.0)),
            half_angle_deg=float(bathy_cfg.get("half_angle_deg", 35.0)),
            update_rate_hz=float(bathy_cfg.get("update_rate_hz", 2.0)),
            range_sigma_m=float(bathy_cfg.get("range_sigma_m", 0.35)),
            dropout_prob=float(bathy_cfg.get("dropout_prob", 0.0)),
            min_altitude_m=float(bathy_cfg.get("min_altitude_m", 0.5)),
            max_altitude_m=float(bathy_cfg.get("max_altitude_m", 100.0)),
            debug_print=bool(bathy_cfg.get("debug_print", False)),
        )

        dvl_active = bool(state.nav_debug.get("dvl_accepted", False))
        search_radius_m = float(est_cfg.get("search_radius_m", 40.0))
        search_step_m = float(est_cfg.get("search_step_m", 4.0))
        blend_gain = float(est_cfg.get("blend_gain", 0.35))
        max_step_m = float(est_cfg.get("max_correction_step_m", 3.0))
        imu_drift_attenuation_max = float(est_cfg.get("imu_drift_attenuation_max", 0.65))
        if dvl_active:
            blend_gain = min(blend_gain, 0.2)
            max_step_m = min(max_step_m, 1.5)
            imu_drift_attenuation_max = min(imu_drift_attenuation_max, 0.2)
            search_radius_m = min(search_radius_m, 15.0)
            search_step_m = max(search_step_m, 5.0)

        corr_cfg = BathyCorrectionConfig(
            enabled=True,
            search_radius_m=search_radius_m,
            search_step_m=search_step_m,
            blend_gain=blend_gain,
            max_step_m=max_step_m,
            min_valid_returns=int(est_cfg.get("min_valid_beams", 8)),
            min_texture_m=float(est_cfg.get("min_texture_m", 0.75)),
            max_texture_m=float(est_cfg.get("max_texture_m", 8.0)),
            min_gradient_norm=float(est_cfg.get("min_gradient_norm", 0.01)),
            max_rmse_m=float(est_cfg.get("max_score_threshold", 3.0)),
            min_rmse_improvement_m=float(est_cfg.get("min_score_improvement", 0.25)),
            imu_drift_attenuation_max=imu_drift_attenuation_max,
            confidence_smoothing=float(est_cfg.get("confidence_smoothing", 0.85)),
        )

        bathy_result = bathy_update_step(
            grid=grid,
            estimate=state.estimate,
            truth=state.truth,
            drift_xy_m=(state.estimate.drift_err_x, state.estimate.drift_err_y),
            yaw_rad=float(state.truth.psi),
            sensor_cfg=sensor_cfg,
            corr_cfg=corr_cfg,
            prev_confidence=float(state.estimate.bathy_confidence),
            rng=state.rng,
        )

        meas = bathy_result.get("measurement")
        update = bathy_result.get("update", {})
        conf = float(bathy_result.get("confidence", 0.0))
        corr_xy = np.asarray(update.get("correction_xy_m", [0.0, 0.0]), dtype=float)

        state.last_bathy_measurement = meas.as_dict() if meas is not None else None

        state.estimate.drift_err_x = float(state.estimate.x - state.truth.x)
        state.estimate.drift_err_y = float(state.estimate.y - state.truth.y)
        state.estimate.drift_err_z = float(state.estimate.z - state.truth.z)

        state.estimate.bathy_confidence = conf
        state.estimate.bathy_correction_x = float(corr_xy[0])
        state.estimate.bathy_correction_y = float(corr_xy[1])
        state.estimate.bathy_correction_z = 0.0

        state.bathy_debug = {
            "enabled": True,
            "status": update.get("status", "unknown"),
            "update_applied": int(bool(update.get("accepted", False))),
            "confidence": conf,
            "valid_returns": int(update.get("valid_returns", 0)),
            "texture_m": float(update.get("texture_m", 0.0)),
            "gradient_norm": float(update.get("gradient_norm", 0.0)),
            "best_score": float(update.get("best_score", float("nan"))) if "best_score" in update else float("nan"),
            "prior_score": float(update.get("prior_score", float("nan"))) if "prior_score" in update else float("nan"),
            "score_improvement": float(update.get("score_improvement", 0.0)),
            "correction_x": float(corr_xy[0]),
            "correction_y": float(corr_xy[1]),
        }
        state.nav_debug.update({
            "bathy_enabled": True,
            "bathy_valid_beams": state.bathy_debug.get("valid_returns", 0),
            "bathy_used": True,
            "bathy_accepted": bool(update.get("accepted", False)),
            "bathy_lock": bool(update.get("accepted", False)),
            "bathy_best_score": state.bathy_debug.get("best_score", ""),
            "bathy_prior_score": state.bathy_debug.get("prior_score", ""),
            "bathy_score_improvement": state.bathy_debug.get("score_improvement", 0.0),
            "bathy_gradient_norm": state.bathy_debug.get("gradient_norm", 0.0),
            "bathy_best_x": update.get("best_xy_m", ["", ""])[0] if isinstance(update.get("best_xy_m"), (list, tuple, np.ndarray)) and len(update.get("best_xy_m")) >= 2 else "",
            "bathy_best_y": update.get("best_xy_m", ["", ""])[1] if isinstance(update.get("best_xy_m"), (list, tuple, np.ndarray)) and len(update.get("best_xy_m")) >= 2 else "",
            "bathy_corrected_x": float(state.estimate.x),
            "bathy_corrected_y": float(state.estimate.y),
            "bathy_center_elev_m": float(getattr(meas, "altitude_m", float("nan"))) if meas is not None else "",
            "bathy_status": update.get("status", "unknown"),
            "bathy_innovation_xy_m": float(np.linalg.norm(update.get("innovation_xy_m", [0.0, 0.0]))) if update.get("innovation_xy_m") is not None else "",
            "correction_x": float(corr_xy[0]),
            "correction_y": float(corr_xy[1]),
        })

        if state.bathy_debug.get("update_applied", 0):
            print(
                f"[{state.name}] BATHY APPLIED "
                f"t={self.time_s:.1f} "
                f"status={state.bathy_debug.get('status')} "
                f"conf={state.bathy_debug.get('confidence', 0.0):.2f} "
                f"corr=({state.bathy_debug.get('correction_x', 0.0):.2f}, "
                f"{state.bathy_debug.get('correction_y', 0.0):.2f}) "
                f"drift=({state.estimate.drift_err_x:.2f}, {state.estimate.drift_err_y:.2f})"
            )
        else:
            altitude = float(getattr(meas, "altitude_m", float("nan"))) if meas is not None else float("nan")

            print(
                f"[{state.name}] BATHY SKIP "
                f"t={self.time_s:.1f} "
                f"status={state.bathy_debug.get('status')} "
                f"altitude={altitude:.1f} "
                f"conf={state.bathy_debug.get('confidence', 0.0):.2f} "
                f"texture={state.bathy_debug.get('texture_m', 0.0):.2f} "
                f"valid={state.bathy_debug.get('valid_returns', 0)}"
            )

    def _step_attached_vehicle(self, state: VehicleState) -> None:
        if not state.parent_vehicle or state.parent_vehicle not in self.vehicle_states:
            return
        parent = self.vehicle_states[state.parent_vehicle]
        state.pos = [
            parent.pos[0] + state.attachment_offset[0],
            parent.pos[1] + state.attachment_offset[1],
            parent.pos[2] + state.attachment_offset[2],
        ]
        state.truth.x = state.pos[0]
        state.truth.y = state.pos[1]
        state.truth.z = state.pos[2]
        state.estimate.x = state.pos[0]
        state.estimate.y = state.pos[1]
        state.estimate.z = state.pos[2]

    def _current_target(self, state: VehicleState) -> Optional[Tuple[float, float, float]]:
        if state.current_index is None or state.current_index >= len(state.route):
            return None
        route_item = state.route[state.current_index]
        if state.current_point_index >= len(route_item.points):
            return None
        return route_item.points[state.current_point_index]

    def _advance_route(self, state: VehicleState) -> None:
        
        print(f"[{state.name}] advancing from route index {state.current_index}")
        if state.current_index is not None and state.current_index < len(state.route):
            route_item = state.route[state.current_index]
            print(
                f"[{state.name}] completed route_id={route_item.route_id} "
                f"emit_event={route_item.emit_event} "
                f"on_complete={route_item.on_complete}"
    )
        
        if state.current_index is None:
            state.current_index = 0
            state.current_trigger_executed = False
            state.current_point_index = 0
            return
        route_item = state.route[state.current_index]
        state.completed_once.add(route_item.route_id)
        self._apply_actions(state, route_item.on_complete)
        if route_item.emit_event:
            self._emit_event(route_item.emit_event)
        state.current_index += 1
        state.current_trigger_executed = False
        state.current_point_index = 0
        state.waiting_since = None
        if state.current_index >= len(state.route):
            state.done = True

        print(
            f"[{state.name}] new route index={state.current_index} done={state.done}"
        )
        if state.current_index is not None and state.current_index < len(state.route):
            nxt = state.route[state.current_index]
            print(
                f"[{state.name}] next route_id={nxt.route_id} "
                f"trigger={nxt.trigger} "
                f"points={len(nxt.points)}"
            )
            print(
                f"[{state.name}] ACTIVATED route_id={nxt.route_id} "
                f"target0={nxt.points[0] if nxt.points else None} "
                f"truth_now=({state.truth.x:.1f}, {state.truth.y:.1f}, {state.truth.z:.1f})"
            )

    def _activate_current_route(self, state: VehicleState) -> Optional[RouteItem]:
        if state.done:
            return None
        if state.current_index is None:
            state.current_index = 0
        if state.current_index >= len(state.route):
            state.done = True
            return None
        route_item = state.route[state.current_index]
        if not state.current_trigger_executed:
            if self._trigger_condition_met(route_item.trigger, state):
                state.current_trigger_executed = True
                self._apply_actions(state, route_item.on_trigger)
                if route_item.emit_event and not route_item.on_complete:
                    self._emit_event(route_item.emit_event)
            else:
                
                print(
                    f"[{state.name}] waiting on trigger for route_id={route_item.route_id}: "
                    f"{route_item.trigger} at sim_t={self.time_s:.1f}"
                )

                return None
        return route_item

    def step(self) -> bool:
        active_any = False
        for state in self.vehicle_states.values():
            if state.done:
                continue
            active_any = True

            route_item = self._activate_current_route(state)

            if state.attached:
                if route_item is None:
                    self._step_attached_vehicle(state)
                    continue
                # If the trigger fired, on_trigger may have detached the vehicle.
                if state.attached:
                    self._step_attached_vehicle(state)
                    continue

            if route_item is None:
                continue

            target = self._current_target(state)
            if target is None:
                self._advance_route(state)
                continue

            if self.time_s < 5.0 or abs(self.time_s % 100.0) < 1e-9:
                print(f"[{state.name}] route_id={state.route[state.current_index].route_id}")
                print(
                    f"[{state.name}] truth=({state.truth.x:.1f}, {state.truth.y:.1f}, {state.truth.z:.1f})"
                )
                print(f"[{state.name}] target={target}")
                print(
                    f"[{state.name}] attached={state.attached}, "
                    f"parent={state.parent_vehicle}, offset={state.attachment_offset}"
                )
            
            runtime_cfg = state.runtime_cfg
            active_route = state.route[state.current_index]

            route_speed = float(
                active_route.params.get(
                    "leg_speed_mps",
                    runtime_cfg.get("default_speed_mps", 1.0),
                )
            )

            state.current_target_xyz = (target[0], target[1], target[2], route_speed)            
            
            control_dt = float(
                runtime_cfg.get(
                    "control_dt_s",
                    self.sim_cfg.get("global", {}).get("control_dt_s", self.time_step_s),
                )
            )            
            log_dt = float(self.sim_cfg.get("logging", {}).get("sample_interval_s", self.time_step_s))
            sensors_cfg = runtime_cfg.get("sensors", {})
            dvl_cfg = sensors_cfg.get("dvl", {})
            bathy_cfg = sensors_cfg.get("bathymetry_cone", {})
            dvl_rate_hz = float(dvl_cfg.get("update_rate_hz", 5.0)) if bool(dvl_cfg.get("enabled", False)) else 0.0
            bathy_rate_hz = float(bathy_cfg.get("update_rate_hz", 0.5)) if bool(bathy_cfg.get("enabled", False)) else 0.0
            dvl_dt = (1.0 / max(dvl_rate_hz, 1e-6)) if dvl_rate_hz > 0.0 else None
            bathy_dt = (1.0 / max(bathy_rate_hz, 1e-6)) if bathy_rate_hz > 0.0 else None

            env_forcing = self._get_env_forcing(runtime_cfg, self.time_s)
            if self.time_s + 1e-9 >= state.next_control_time_s:
                controller_uses_estimate = bool(runtime_cfg.get("controller_uses_estimate", True))
                nav_state = state.estimate if controller_uses_estimate else truth_to_estimate(state.truth)
                
                if state.name == "ssv_1" and self.time_s < 20.0:
                    print(
                        f"[{state.name}] GUIDANCE INPUT "
                        f"t={self.time_s:.1f} "
                        f"truth=({state.truth.x:.2f},{state.truth.y:.2f},{state.truth.z:.2f}) "
                        f"psi={state.truth.psi:.3f} u={state.truth.u:.3f} "
                        f"target={state.current_target_xyz}"
                    )
                
                state.command = compute_guidance_command(
                    runtime_cfg.get("controller", "direct_pursuit"),
                    state.truth,
                    state.estimate,
                    state.current_target_xyz,
                    runtime_cfg.get("gains", {}),
                    runtime_cfg.get("coeffs", {}),
                    bool(runtime_cfg.get("controller_uses_estimate", True)),
                    bool(runtime_cfg.get("depth_channel_enabled", False)),
                )

                if state.name == "ssv_1" and self.time_s < 20.0:
                    print(f"[{state.name}] GUIDANCE OUTPUT t={self.time_s:.1f} cmd={state.command}")

                state.next_control_time_s += control_dt

                if int(self.time_s) % 100 == 0 and abs(self.time_s - round(self.time_s)) < 1e-9:
                    print(
                        f"[{state.name}] t={self.time_s:.1f} "
                        f"target={target} "
                        f"cmd={state.command}"
    )



            if int(self.time_s) % 100 == 0 and abs(self.time_s - round(self.time_s)) < 1e-9:
                print(
                    f"[{state.name}] post-step "
                    f"x={state.truth.x:.2f} y={state.truth.y:.2f} z={state.truth.z:.2f} "
                    f"u={state.truth.u:.2f} psi={state.truth.psi:.2f}"
                )   

            state.pos = [state.truth.x, state.truth.y, state.truth.z]
            prev_truth = TruthState(
                t=state.truth.t,
                x=state.truth.x,
                y=state.truth.y,
                z=state.truth.z,
                u=state.truth.u,
                psi=state.truth.psi,
                r=state.truth.r,
                w=state.truth.w,
                theta=state.truth.theta,
                q=state.truth.q,
                delta_act=state.truth.delta_act,
                elev_act=state.truth.elev_act,
                distance_traveled_m=state.truth.distance_traveled_m,
            )

            if state.name == "ssv_1":
                if not np.isfinite(state.truth.x) or not np.isfinite(state.truth.y) or not np.isfinite(state.truth.psi):
                    print(
                        f"[{state.name}] PRE-PLANT NAN "
                        f"t={self.time_s:.1f} "
                        f"truth={state.truth} "
                        f"cmd={state.command}"
                    )
                    raise RuntimeError("SSV truth became non-finite before plant step")
            if state.name == "ssv_1" and self.time_s > 3535:
                print(
                    f"[{state.name}] PRE step t={self.time_s:.1f} "
                    f"x={state.truth.x:.3f} y={state.truth.y:.3f} "
                    f"psi={state.truth.psi:.6f} r={state.truth.r:.6f} "
                    f"u={state.truth.u:.6f} delta_act={state.truth.delta_act:.6f} "
                    f"cmd_thrust={state.command.thrust:.6f} cmd_delta={state.command.delta_cmd:.6f}"
                )
            state.truth = step_vehicle_plant(
                state.truth,
                state.command,
                runtime_cfg.get("coeffs", {}),
                env_forcing,
                self.time_step_s,
                runtime_cfg.get("vehicle_mode", "surface"),
                bool(runtime_cfg.get("depth_channel_enabled", False)),
            )

            if state.name == "ssv_1":
                if not np.isfinite(state.truth.x) or not np.isfinite(state.truth.y) or not np.isfinite(state.truth.psi):
                    print(
                        f"[{state.name}] POST-PLANT NAN "
                        f"t={self.time_s:.1f} "
                        f"truth={state.truth} "
                        f"cmd={state.command} "
                        f"target={target}"
                    )
                    raise RuntimeError("SSV truth became non-finite after plant step")
                if state.name == "ssv_1" and self.time_s > 3535:
                    print(
                        f"[{state.name}] POST step t={self.time_s:.1f} "
                        f"x={state.truth.x:.3f} y={state.truth.y:.3f} "
                        f"psi={state.truth.psi:.6f} r={state.truth.r:.6f} "
                        f"u={state.truth.u:.6f} delta_act={state.truth.delta_act:.6f}"
                    )
            state.pos = [state.truth.x, state.truth.y, state.truth.z]

            state.estimate = propagate_estimate(
                state.estimate,
                prev_truth,
                state.truth,
                runtime_cfg.get("drift", {}),
                self.time_step_s,
                state.rng,
            )

            if dvl_dt is not None and self.time_s + 1e-9 >= state.next_dvl_time_s:
                self._apply_dvl_localization(state, dvl_dt)
                state.next_dvl_time_s += dvl_dt

            if bathy_dt is not None and self.time_s + 1e-9 >= state.next_bathy_time_s:
                self._apply_bathymetry_localization(state, bathy_dt)
                state.next_bathy_time_s += bathy_dt





            if self.time_s + 1e-9 >= state.next_log_time_s:
                state.log_rows.append(
                    make_vehicle_log_row(
                        self.time_s,
                        state.name,
                        state.truth,
                        state.estimate,
                        state.command,
                        state.current_target_xyz[:3],
                        bool(runtime_cfg.get("depth_channel_enabled", False)),
                        nav_debug={**state.nav_debug, **state.bathy_debug},
                    )
                )
                state.next_log_time_s += log_dt

                if self.time_s < 5.0 or int(self.time_s) % 100 == 0:
                    print(f"[{state.name}] logging at t={self.time_s:.1f}, rows={len(state.log_rows)}")

            acceptance_mode = str(runtime_cfg.get("waypoint_acceptance", self.sim_cfg.get("global", {}).get("waypoint_acceptance_default", "truth"))).lower()
            acceptance_state = state.estimate if acceptance_mode == "estimated" else truth_to_estimate(state.truth)
            
            dist_xy = ((target[0] - state.truth.x) ** 2 + (target[1] - state.truth.y) ** 2) ** 0.5
            dist_z = abs(target[2] - state.truth.z)

            if int(self.time_s) % 50 == 0 or dist_xy < 20:
                print(
                    f"[{state.name}] reach_check route_id={state.route[state.current_index].route_id} "
                    f"truth=({state.truth.x:.1f},{state.truth.y:.1f},{state.truth.z:.1f}) "
                    f"target=({target[0]:.1f},{target[1]:.1f},{target[2]:.1f}) "
                    f"dist_xy={dist_xy:.1f} dist_z={dist_z:.1f}"
    )
            
            if waypoint_reached(
                acceptance_mode,
                state.truth,
                state.estimate,
                state.current_target_xyz,
                runtime_cfg.get("coeffs", {}),
                bool(runtime_cfg.get("depth_channel_enabled", False)),
            ):
                current_item = state.route[state.current_index] if state.current_index is not None else None

                if current_item is None:
                    pass
                elif state.current_point_index < len(current_item.points) - 1:
                    state.current_point_index += 1
                else:
                    # Final point of this route item reached
                    route_id = current_item.route_id

                    # Loiter behavior: do not complete until event/time condition is satisfied
                    if str(current_item.mode).lower() == "loiter":
                        if state.waiting_since is None:
                            state.waiting_since = self.time_s

                        loiter_until_event = current_item.params.get("loiter_until_event")
                        loiter_seconds = float(current_item.params.get("loiter_seconds", 0.0))

                        event_satisfied = bool(loiter_until_event) and (loiter_until_event in self.event_names)
                        time_satisfied = loiter_seconds > 0.0 and ((self.time_s - state.waiting_since) >= loiter_seconds)

                        if event_satisfied or time_satisfied:
                            state.completed_waypoints.add(route_id)
                            self._advance_route(state)
                    else:
                        state.completed_waypoints.add(route_id)
                        self._advance_route(state)

            if self.time_s + 1e-9 >= state.next_log_time_s:
                state.log_rows.append(
                    make_vehicle_log_row(
                        self.time_s,
                        state.name,
                        state.truth,
                        state.estimate,
                        state.command,
                        state.current_target_xyz[:3],
                        bool(runtime_cfg.get("depth_channel_enabled", False)),
                        nav_debug={**state.nav_debug, **state.bathy_debug},
                    )
                )
                state.next_log_time_s += log_dt

        self._sync_attached_vehicles()
        self._snapshot()
        self.time_s += self.time_step_s
        return active_any

    def run(self, max_time_s: float) -> None:
        while self.time_s <= max_time_s and self.step():
            pass
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging_cfg = self.sim_cfg.get("logging", {})
        for vehicle_name, state in self.vehicle_states.items():
            print(vehicle_name)
            print("  done:", state.done)
            print("  truth history len:", len(state.truth_history.get("x", [])))
            print("  estimate history len:", len(state.estimate_history.get("x", [])))
            print("  log rows:", len(state.log_rows))
            print("  final truth:", state.truth)
        
        if logging_cfg.get("enabled", True) and logging_cfg.get("one_csv_per_vehicle", True):
            for vehicle_name, state in self.vehicle_states.items():
                csv_path = self.output_dir / f"{vehicle_name}.csv"
                write_vehicle_csv(state.log_rows, csv_path)


def load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at root of {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate and validate Poseidon mission scenarios")
    parser.add_argument("scenario", type=Path, help="Path to scenario YAML file")
    parser.add_argument("--sim-config", type=Path, default=DEFAULT_SIM_CONFIG_PATH, help="Path to sim_config.yaml")
    parser.add_argument("--dt", type=float, default=DEFAULT_TIME_STEP_S, help="Simulation time step [s]")
    parser.add_argument("--max-time", type=float, default=DEFAULT_MAX_SIM_TIME_S, help="Maximum simulation duration [s]")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override log output directory")
    parser.add_argument("--no-plot", action="store_true", help="Skip static plots")
    parser.add_argument("--no-animate", action="store_true", help="Skip animation")
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Animation playback speed multiplier (e.g. 2.0 = 2x faster)",
    )
    
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    scenario = load_yaml_file(args.scenario)
    sim_cfg = load_sim_config(args.sim_config)
    animator = ScenarioAnimator(
        scenario=scenario,
        sim_cfg=sim_cfg,
        time_step_s=args.dt,
        output_dir=args.output_dir,
    )
    animator.run(max_time_s=args.max_time)

    viz_cfg = sim_cfg.get("visualization", {})
    truth_histories = {
        name: state.truth_history
        for name, state in animator.vehicle_states.items()
    }

    if not args.no_plot:
        if viz_cfg.get("enable_2d", True):
            plot_mission_2d(animator.planned_paths, truth_histories)
        if viz_cfg.get("enable_3d", True):
            plot_mission_3d(animator.planned_paths, truth_histories)

    anim2d = None
    anim3d = None

    if not args.no_animate:
        if viz_cfg.get("enable_2d", True):
            anim2d = animate_mission_2d(
                animator.planned_paths,
                truth_histories,
                playback_speed=args.playback_speed,
            )
        if viz_cfg.get("enable_3d", True):
            anim3d = animate_mission_3d(
                animator.planned_paths,
                truth_histories,
                playback_speed=args.playback_speed,
            )
    plt.show()


if __name__ == "__main__":
    main()


