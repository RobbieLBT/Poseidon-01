from __future__ import annotations

import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install it with: pip install pyyaml") from exc


# ============================================================
# Dataclasses
# ============================================================


@dataclass
class TruthState:
    t: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    u: float = 0.0
    psi: float = 0.0
    r: float = 0.0
    w: float = 0.0
    theta: float = 0.0
    q: float = 0.0
    delta_act: float = 0.0
    elev_act: float = 0.0
    distance_traveled_m: float = 0.0


@dataclass
class EstimateState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    psi: float = 0.0
    theta: float = 0.0
    phi: float = 0.0
    quat_w: float = 1.0
    quat_x: float = 0.0
    quat_y: float = 0.0
    quat_z: float = 0.0
    gyro_bias_x: float = 0.0
    gyro_bias_y: float = 0.0
    gyro_bias_z: float = 0.0
    accel_bias_x: float = 0.0
    accel_bias_y: float = 0.0
    accel_bias_z: float = 0.0
    current_bias_x: float = 0.0
    current_bias_y: float = 0.0
    current_bias_z: float = 0.0
    cov_pos_trace: float = 0.0
    cov_vel_trace: float = 0.0
    cov_att_trace: float = 0.0
    heading_rw: float = 0.0
    cov_trace: float = 0.0
    health_status: str = ""

    drift_err_x: float = 0.0
    drift_err_y: float = 0.0
    drift_err_z: float = 0.0
    drift_err_psi: float = 0.0

    bathy_confidence: float = 0.0
    bathy_correction_x: float = 0.0
    bathy_correction_y: float = 0.0
    bathy_correction_z: float = 0.0


@dataclass
class Command:
    thrust: float = 0.0
    delta_cmd: float = 0.0
    elev_cmd: float = 0.0
    buoyancy_cmd: float = 0.0


from dataclasses import dataclass, field

@dataclass
class EnvForcing:
    current_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    current_z: float = 0.0

class ForcingProvider:
    def forcing_at_time(self, t_s: float, vehicle_mode: str) -> EnvForcing:
        raise NotImplementedError


# ============================================================
# Config helpers
# ============================================================


def load_sim_config(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Simulation config must parse to a dictionary")
    cfg.setdefault("__source_path__", str(path.resolve()))
    return cfg


def get_global_setting(sim_cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    return sim_cfg.get("global", {}).get(key, default)


def get_vehicle_runtime_config(sim_cfg: Dict[str, Any], vehicle_id: str) -> Dict[str, Any]:
    vehicles = sim_cfg.get("vehicles", {})
    profiles = sim_cfg.get("profiles", {})
    if vehicle_id not in vehicles:
        raise KeyError(f"Vehicle {vehicle_id} missing from sim config")
    vehicle_cfg = dict(vehicles[vehicle_id])
    profile_name = vehicle_cfg.get("sim_profile")
    if not profile_name:
        raise ValueError(f"Vehicle {vehicle_id} missing sim_profile")
    if profile_name not in profiles:
        raise KeyError(f"Unknown sim profile {profile_name} for vehicle {vehicle_id}")
    runtime = dict(profiles[profile_name])
    runtime.update(vehicle_cfg)
    runtime["sim_profile"] = profile_name
    runtime.setdefault(
        "waypoint_acceptance",
        get_global_setting(sim_cfg, "waypoint_acceptance_default", "estimated"),
    )
    runtime.setdefault(
        "playback_mode",
        get_global_setting(sim_cfg, "playback_mode_default", "dynamic"),
    )
    return runtime


# ============================================================
# Basic helpers
# ============================================================


def wrap_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def sat(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def norm2(vec: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(vec, dtype=float)))


def deg2rad_if_present(value_deg: Optional[float], default_rad: float = 0.0) -> float:
    if value_deg is None:
        return default_rad
    return math.radians(float(value_deg))


def truth_to_estimate(truth: TruthState) -> EstimateState:
    return EstimateState(
        x=truth.x,
        y=truth.y,
        z=truth.z,
        vx=truth.u * math.cos(truth.psi),
        vy=truth.u * math.sin(truth.psi),
        vz=truth.w,
        psi=truth.psi,
        theta=truth.theta,
        phi=0.0,
        heading_rw=0.0,
        drift_err_x=0.0,
        drift_err_y=0.0,
        drift_err_z=0.0,
        drift_err_psi=0.0,
    )


def initial_truth_from_pose(pose_xyz: Sequence[float]) -> TruthState:
    return TruthState(x=float(pose_xyz[0]), y=float(pose_xyz[1]), z=float(pose_xyz[2]))


# ============================================================
# Environment
# ============================================================


def get_current_xy(env_cfg: Dict[str, Any], vehicle_mode: str) -> np.ndarray:
    if vehicle_mode == "surface":
        if env_cfg.get("surface_current_enabled", False):
            return np.asarray(env_cfg.get("surface_current_xy", [0.0, 0.0]), dtype=float)
        return np.zeros(2, dtype=float)

    if env_cfg.get("underwater_current_enabled", False):
        return np.asarray(env_cfg.get("underwater_current_xy", [0.0, 0.0]), dtype=float)
    return np.zeros(2, dtype=float)


def get_current_z(env_cfg: Dict[str, Any], vehicle_mode: str) -> float:
    if vehicle_mode == "underwater" and env_cfg.get("underwater_current_z_enabled", False):
        return float(env_cfg.get("underwater_current_z", 0.0))
    return 0.0


def get_environment_forcing(env_cfg: Dict[str, Any], vehicle_mode: str) -> EnvForcing:
    return EnvForcing(
        current_xy=get_current_xy(env_cfg, vehicle_mode),
        current_z=get_current_z(env_cfg, vehicle_mode),
    )


# ============================================================
# Controllers
# ============================================================


def compute_vertical_command(
    ctrl_z: float,
    theta: float,
    q: float,
    target_z: float,
    gains: Dict[str, float],
    coeffs: Dict[str, float],
    depth_enabled: bool,
) -> Tuple[float, float]:
    if not depth_enabled:
        return 0.0, gains.get("B_trim", 0.0)

    k_z = float(gains.get("K_z", 0.0))
    k_theta = float(gains.get("K_theta", 0.0))
    k_q = float(gains.get("K_q", 0.0))
    elev_max = deg2rad_if_present(coeffs.get("elev_max_deg"), coeffs.get("elev_max", math.radians(18.0)))

    e_z = target_z - ctrl_z
    theta_cmd = sat(k_z * e_z, -math.radians(15.0), math.radians(15.0))
    e_theta = wrap_angle(theta_cmd - theta)
    elev_cmd = sat(k_theta * e_theta - k_q * q, -elev_max, elev_max)
    return elev_cmd, float(gains.get("B_trim", 0.0))



def compute_approach_target_speed(
    cruise_speed: float,
    dx: float,
    dy: float,
    coeffs: Dict[str, float],
) -> float:
    slowdown_radius_m = float(coeffs.get("slowdown_radius_m", 0.0))
    min_approach_speed_mps = float(coeffs.get("min_approach_speed_mps", 0.0))

    if slowdown_radius_m <= 0.0:
        return max(0.0, cruise_speed)

    dist_xy = math.hypot(dx, dy)
    if dist_xy >= slowdown_radius_m:
        return max(0.0, cruise_speed)

    frac = max(0.0, min(1.0, dist_xy / slowdown_radius_m))
    tapered_speed = cruise_speed * frac
    return max(0.0, min(cruise_speed, max(min_approach_speed_mps, tapered_speed)))



def compute_direct_pursuit_command(
    nav_x: float,
    nav_y: float,
    nav_z: float,
    nav_psi: float,
    truth_u: float,
    truth_r: float,
    truth_theta: float,
    truth_q: float,
    target_xyz: Sequence[float],
    gains: Dict[str, float],
    coeffs: Dict[str, float],
    depth_enabled: bool,
) -> Command:
    dx = float(target_xyz[0]) - nav_x
    dy = float(target_xyz[1]) - nav_y
    psi_cmd = math.atan2(dy, dx)
    e_psi = wrap_angle(psi_cmd - nav_psi)

    delta_max = deg2rad_if_present(coeffs.get("delta_max_deg"), coeffs.get("delta_max", math.radians(20.0)))
    thrust_max = float(coeffs.get("T_max", 7.0))
    cruise_speed = float(target_xyz[3]) if len(target_xyz) > 3 else 0.0
    target_speed = compute_approach_target_speed(cruise_speed, dx, dy, coeffs)

    thrust = sat(float(gains.get("K_u", 0.0)) * (target_speed - truth_u), 0.0, thrust_max)
    delta_cmd = sat(float(gains.get("K_psi", 0.0)) * e_psi, -delta_max, delta_max)
    elev_cmd, buoyancy_cmd = compute_vertical_command(
        ctrl_z=nav_z,
        theta=truth_theta,
        q=truth_q,
        target_z=float(target_xyz[2]),
        gains=gains,
        coeffs=coeffs,
        depth_enabled=depth_enabled,
    )
    return Command(thrust=thrust, delta_cmd=delta_cmd, elev_cmd=elev_cmd, buoyancy_cmd=buoyancy_cmd)


def compute_damped_pursuit_command(
    nav_x: float,
    nav_y: float,
    nav_z: float,
    nav_psi: float,
    truth_u: float,
    truth_r: float,
    truth_theta: float,
    truth_q: float,
    target_xyz: Sequence[float],
    gains: Dict[str, float],
    coeffs: Dict[str, float],
    depth_enabled: bool,
) -> Command:
    cmd = compute_direct_pursuit_command(
        nav_x=nav_x,
        nav_y=nav_y,
        nav_z=nav_z,
        nav_psi=nav_psi,
        truth_u=truth_u,
        truth_r=truth_r,
        truth_theta=truth_theta,
        truth_q=truth_q,
        target_xyz=target_xyz,
        gains=gains,
        coeffs=coeffs,
        depth_enabled=depth_enabled,
    )
    delta_max = deg2rad_if_present(coeffs.get("delta_max_deg"), coeffs.get("delta_max", math.radians(20.0)))
    cmd.delta_cmd = sat(cmd.delta_cmd - float(gains.get("K_r", 0.0)) * truth_r, -delta_max, delta_max)
    return cmd


def compute_guidance_command(
    controller_name: str,
    truth: TruthState,
    estimate: EstimateState,
    target_xyz: Sequence[float],
    gains: Dict[str, float],
    coeffs: Dict[str, float],
    controller_uses_estimate: bool,
    depth_enabled: bool,
) -> Command:
    nav_x = estimate.x if controller_uses_estimate else truth.x
    nav_y = estimate.y if controller_uses_estimate else truth.y
    nav_z = estimate.z if controller_uses_estimate else truth.z
    nav_psi = estimate.psi if controller_uses_estimate else truth.psi

    if controller_name == "direct_pursuit":
        return compute_direct_pursuit_command(
            nav_x, nav_y, nav_z, nav_psi,
            truth.u, truth.r, truth.theta, truth.q,
            target_xyz, gains, coeffs, depth_enabled,
        )
    if controller_name == "damped_pursuit":
        return compute_damped_pursuit_command(
            nav_x, nav_y, nav_z, nav_psi,
            truth.u, truth.r, truth.theta, truth.q,
            target_xyz, gains, coeffs, depth_enabled,
        )
    raise ValueError(f"Unknown controller_name: {controller_name}")


# ============================================================
# Plant models
# ============================================================


def step_surface_plant(
    truth: TruthState,
    command: Command,
    coeffs: Dict[str, float],
    forcing: EnvForcing,
    dt: float,
) -> TruthState:
    a_u = float(coeffs.get("a_u", 0.18))
    b_t = float(coeffs.get("b_T", 0.28))
    a_r = float(coeffs.get("a_r", 0.80))
    b_du = float(coeffs.get("b_du", 0.22))
    tau_delta = max(1e-6, float(coeffs.get("tau_delta", 0.70)))
    delta_max = math.radians(float(coeffs.get("delta_max_deg", 30.0)))

    d_delta = (command.delta_cmd - truth.delta_act) / tau_delta
    du = -a_u * truth.u + b_t * command.thrust
    dr = -a_r * truth.r + b_du * truth.u * truth.delta_act
    dpsi = truth.r

    delta_act_next = truth.delta_act + dt * d_delta
    if not math.isfinite(delta_act_next):
        delta_act_next = 0.0
    delta_act_next = max(-delta_max, min(delta_max, delta_act_next))

    u_next = max(0.0, truth.u + dt * du)

    r_next = truth.r + dt * dr
    if not math.isfinite(r_next):
        r_next = 0.0

    psi_next = wrap_angle(truth.psi + dt * dpsi)
    if not math.isfinite(psi_next):
        psi_next = 0.0

    body_vel_xy = np.array(
        [u_next * math.cos(psi_next), u_next * math.sin(psi_next)],
        dtype=float,
    )
    ground_vel_xy = body_vel_xy + forcing.current_xy
    step_xy = dt * ground_vel_xy

    return TruthState(
        t=truth.t + dt,
        x=truth.x + float(step_xy[0]),
        y=truth.y + float(step_xy[1]),
        z=truth.z,
        u=u_next,
        psi=psi_next,
        r=r_next,
        w=0.0,
        theta=0.0,
        q=0.0,
        delta_act=delta_act_next,
        elev_act=0.0,
        distance_traveled_m=truth.distance_traveled_m + norm2(step_xy),
    )


def step_auv_plant(
    truth: TruthState,
    command: Command,
    coeffs: Dict[str, float],
    forcing: EnvForcing,
    dt: float,
    depth_enabled: bool,
) -> TruthState:
    horizontal = step_surface_plant(truth, command, coeffs, forcing, dt)
    if not depth_enabled:
        return horizontal

    a_w = float(coeffs.get("a_w", 0.65))
    b_B = float(coeffs.get("b_B", 0.90))
    b_theta = float(coeffs.get("b_theta", 0.35))
    a_theta = float(coeffs.get("a_theta", 0.90))
    b_elev_u = float(coeffs.get("b_elev_u", 0.12))
    tau_elev = float(coeffs.get("tau_elev", 0.60))

    d_elev = (command.elev_cmd - truth.elev_act) / max(tau_elev, 1e-6)
    dq = -a_theta * truth.q + b_elev_u * truth.u * truth.elev_act
    dtheta = truth.q
    dw = -a_w * truth.w + b_B * command.buoyancy_cmd + b_theta * truth.theta
    dz = truth.w + forcing.current_z

    elev_act_next = truth.elev_act + dt * d_elev
    q_next = truth.q + dt * dq
    theta_next = wrap_angle(truth.theta + dt * dtheta)
    w_next = truth.w + dt * dw
    z_next = truth.z + dt * dz
    step_z = dt * dz

    horizontal.z = z_next
    horizontal.w = w_next
    horizontal.theta = theta_next
    horizontal.q = q_next
    horizontal.elev_act = elev_act_next
    horizontal.distance_traveled_m += abs(step_z)
    return horizontal


def step_vehicle_plant(
    truth: TruthState,
    command: Command,
    coeffs: Dict[str, float],
    forcing: EnvForcing,
    dt: float,
    vehicle_mode: str,
    depth_enabled: bool,
) -> TruthState:
    if vehicle_mode == "surface":
        return step_surface_plant(truth, command, coeffs, forcing, dt)
    if vehicle_mode == "underwater":
        return step_auv_plant(truth, command, coeffs, forcing, dt, depth_enabled)
    raise ValueError(f"Unknown vehicle_mode: {vehicle_mode}")


# ============================================================
# Estimate / drift
# ============================================================


def _rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def propagate_estimate(
    prev_est: EstimateState,
    prev_truth: TruthState,
    next_truth: TruthState,
    drift_cfg: Dict[str, Any],
    dt: float,
    rng: Optional[np.random.Generator] = None,
) -> EstimateState:
    if not drift_cfg.get("enabled", False):
        return EstimateState(
            x=next_truth.x,
            y=next_truth.y,
            z=next_truth.z,
            psi=next_truth.psi,
            heading_rw=0.0,
            drift_err_x=0.0,
            drift_err_y=0.0,
            drift_err_z=0.0,
            drift_err_psi=0.0,
        )

    if rng is None:
        rng = _rng_from_seed(int(drift_cfg.get("seed", 7)))

    drift_mode = drift_cfg.get("mode", "percent_step")
    drift_fraction = float(drift_cfg.get("drift_fraction", 0.0))
    heading_bias = math.radians(float(drift_cfg.get("heading_bias_deg", 0.0)))
    sigma_psi = math.radians(float(drift_cfg.get("heading_rw_sigma_deg_sqrt_s", 0.0)))
    sigma_pos = float(drift_cfg.get("pos_rw_sigma_m_sqrt_s", 0.0))
    sigma_z = float(drift_cfg.get("depth_rw_sigma_m_sqrt_s", 0.0))

    true_step_xy = np.array([next_truth.x - prev_truth.x, next_truth.y - prev_truth.y], dtype=float)
    true_step_z = next_truth.z - prev_truth.z

    heading_rw_next = prev_est.heading_rw + sigma_psi * math.sqrt(dt) * float(rng.normal())
    drift_err_psi_next = heading_bias + heading_rw_next
    psi_est_next = wrap_angle(next_truth.psi + drift_err_psi_next)

    prev_drift_xy = np.array([prev_est.drift_err_x, prev_est.drift_err_y], dtype=float)
    prev_drift_z = float(prev_est.drift_err_z)

    if drift_mode == "percent_step":
        step_mag = (1.0 + drift_fraction) * norm2(true_step_xy)
        est_step_xy = step_mag * np.array([math.cos(psi_est_next), math.sin(psi_est_next)], dtype=float)

        predicted_est_xy = np.array([prev_est.x, prev_est.y], dtype=float) + est_step_xy
        truth_xy = np.array([next_truth.x, next_truth.y], dtype=float)
        drift_xy_next = predicted_est_xy - truth_xy
        drift_z_next = (prev_est.z + (1.0 + drift_fraction) * true_step_z) - next_truth.z

    elif drift_mode == "random_walk":
        pos_rw = sigma_pos * math.sqrt(dt) * np.asarray(rng.normal(size=2), dtype=float)
        z_rw = sigma_z * math.sqrt(dt) * float(rng.normal())
        drift_xy_next = prev_drift_xy + pos_rw
        drift_z_next = prev_drift_z + z_rw

    else:
        raise ValueError(f"Unknown drift mode: {drift_mode}")

    est_x = next_truth.x + float(drift_xy_next[0])
    est_y = next_truth.y + float(drift_xy_next[1])
    est_z = next_truth.z + float(drift_z_next)

    return EstimateState(
        x=est_x,
        y=est_y,
        z=est_z,
        psi=psi_est_next,
        heading_rw=heading_rw_next,
        drift_err_x=float(drift_xy_next[0]),
        drift_err_y=float(drift_xy_next[1]),
        drift_err_z=float(drift_z_next),
        drift_err_psi=float(drift_err_psi_next),
    )

def apply_bathy_drift_correction(
    est: EstimateState,
    truth: TruthState,
    correction_x: float,
    correction_y: float,
    correction_z: float,
    confidence: float,
    max_fraction: float = 1.0,
) -> EstimateState:
    gain = max(0.0, min(1.0, confidence * max_fraction))

    drift_err_x = est.drift_err_x - gain * correction_x
    drift_err_y = est.drift_err_y - gain * correction_y
    drift_err_z = est.drift_err_z - gain * correction_z

    return EstimateState(
        **{**est.__dict__},
        x=truth.x + drift_err_x,
        y=truth.y + drift_err_y,
        z=truth.z + drift_err_z,
        drift_err_x=drift_err_x,
        drift_err_y=drift_err_y,
        drift_err_z=drift_err_z,
        bathy_confidence=confidence,
        bathy_correction_x=gain * correction_x,
        bathy_correction_y=gain * correction_y,
        bathy_correction_z=gain * correction_z,
    )

# ============================================================
# Waypoint logic
# ============================================================


_WAYPOINT_CAPTURE_STATE: Dict[Tuple[int, str, Tuple[float, float, float], bool], bool] = {}


def waypoint_distance(state_xyz: Sequence[float], target_xyz: Sequence[float], depth_enabled: bool) -> Tuple[float, float]:
    dx = float(target_xyz[0]) - float(state_xyz[0])
    dy = float(target_xyz[1]) - float(state_xyz[1])
    dz = float(target_xyz[2]) - float(state_xyz[2]) if depth_enabled else 0.0
    return math.hypot(dx, dy), abs(dz)


def _waypoint_hysteresis_ok(
    distance_xy: float,
    distance_z: float,
    wp_radius_xy: float,
    wp_radius_z: float,
    wp_reacquire_radius_xy: float,
    wp_reacquire_radius_z: float,
    capture_key: Tuple[int, str, Tuple[float, float, float], bool],
    depth_enabled: bool,
) -> bool:
    within_capture = (distance_xy < wp_radius_xy) and ((distance_z < wp_radius_z) if depth_enabled else True)
    if within_capture:
        _WAYPOINT_CAPTURE_STATE[capture_key] = True
        return True

    was_captured = _WAYPOINT_CAPTURE_STATE.get(capture_key, False)
    if not was_captured:
        return False

    within_reacquire_deadband = (distance_xy < wp_reacquire_radius_xy) and ((distance_z < wp_reacquire_radius_z) if depth_enabled else True)
    if within_reacquire_deadband:
        return True

    _WAYPOINT_CAPTURE_STATE[capture_key] = False
    return False


def waypoint_reached(
    mode: str,
    truth: TruthState,
    estimate: EstimateState,
    target_xyz: Sequence[float],
    coeffs: Dict[str, float],
    depth_enabled: bool,
) -> bool:
    wp_radius_xy = float(coeffs.get("wp_radius_xy_m", coeffs.get("wp_radius_xy", 6.0)))
    wp_radius_z = float(coeffs.get("wp_radius_z_m", coeffs.get("wp_radius_z", 3.0)))
    wp_reacquire_radius_xy = float(coeffs.get("wp_reacquire_radius_xy_m", coeffs.get("wp_reacquire_radius_xy", max(wp_radius_xy, 1.5 * wp_radius_xy))))
    wp_reacquire_radius_z = float(coeffs.get("wp_reacquire_radius_z_m", coeffs.get("wp_reacquire_radius_z", max(wp_radius_z, 1.5 * wp_radius_z))))

    truth_xy, truth_z = waypoint_distance((truth.x, truth.y, truth.z), target_xyz, depth_enabled)
    est_xy, est_z = waypoint_distance((estimate.x, estimate.y, estimate.z), target_xyz, depth_enabled)

    target_key = (round(float(target_xyz[0]), 3), round(float(target_xyz[1]), 3), round(float(target_xyz[2]), 3))
    truth_key = (id(coeffs), "truth", target_key, depth_enabled)
    est_key = (id(coeffs), "estimated", target_key, depth_enabled)

    truth_ok = _waypoint_hysteresis_ok(
        distance_xy=truth_xy,
        distance_z=truth_z,
        wp_radius_xy=wp_radius_xy,
        wp_radius_z=wp_radius_z,
        wp_reacquire_radius_xy=wp_reacquire_radius_xy,
        wp_reacquire_radius_z=wp_reacquire_radius_z,
        capture_key=truth_key,
        depth_enabled=depth_enabled,
    )
    est_ok = _waypoint_hysteresis_ok(
        distance_xy=est_xy,
        distance_z=est_z,
        wp_radius_xy=wp_radius_xy,
        wp_radius_z=wp_radius_z,
        wp_reacquire_radius_xy=wp_reacquire_radius_xy,
        wp_reacquire_radius_z=wp_reacquire_radius_z,
        capture_key=est_key,
        depth_enabled=depth_enabled,
    )

    if mode == "truth":
        return truth_ok
    if mode == "estimated":
        return est_ok
    if mode == "both":
        return truth_ok and est_ok
    raise ValueError(f"Unknown waypoint acceptance mode: {mode}")


# ============================================================
# Logging
# ============================================================


LOG_COLUMNS = [
    "timestamp_s", "vehicle_id", "row_type", "event_type", "route_id",
    "target_x", "target_y", "target_z",
    "true_x", "true_y", "true_z",
    "est_x", "est_y", "est_z", "est_psi_deg", "est_theta_deg", "est_vx", "est_vy", "est_vz",
    "u", "psi_deg", "r_deg_s", "w", "theta_deg", "q_deg_s",
    "delta_cmd_deg", "delta_act_deg", "elev_cmd_deg", "elev_act_deg", "thrust", "buoyancy_cmd",
    "distance_traveled_m", "waypoint_distance_true_m", "waypoint_distance_est_m",
    "nav_mode", "nav_status", "nav_health", "nav_use_current_state", "nav_pos_std_m", "nav_vel_std_mps", "nav_yaw_std_deg",
    "imu_used",
    "depth_used", "depth_accepted", "depth_innovation_m",
    "compass_used", "compass_accepted", "compass_innovation_deg",
    "dvl_used", "dvl_accepted", "dvl_bottom_lock", "dvl_altitude_m", "dvl_innovation_norm_mps",
    "dvl_vel_blend_gain", "dvl_pos_blend_gain",
    "gps_used", "gps_accepted", "gps_innovation_m",
    "est_current_x", "est_current_y", "est_current_z",
    "bathy_enabled", "bathy_valid_beams", "bathy_used", "bathy_accepted", "bathy_lock",
    "bathy_best_score", "bathy_prior_score", "bathy_score_improvement", "bathy_gradient_norm",
    "bathy_best_x", "bathy_best_y", "bathy_corrected_x", "bathy_corrected_y",
    "bathy_center_elev_m", "bathy_status", "bathy_innovation_xy_m",
    "attached", "parent_vehicle", "notes",
]


def make_vehicle_log_row(
    timestamp_s: float,
    vehicle_id: str,
    truth: TruthState,
    estimate: EstimateState,
    command: Command,
    target_xyz: Sequence[float],
    depth_enabled: bool,
    attached: bool = False,
    parent_vehicle: str = "",
    route_id: str = "",
    notes: str = "",
    nav_debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    true_xy, _ = waypoint_distance((truth.x, truth.y, truth.z), target_xyz, depth_enabled)
    est_xy, _ = waypoint_distance((estimate.x, estimate.y, estimate.z), target_xyz, depth_enabled)
    debug = dict(nav_debug or {})
    return {
        "timestamp_s": round(float(timestamp_s), 3),
        "vehicle_id": vehicle_id,
        "row_type": "sample",
        "event_type": "",
        "route_id": route_id,
        "target_x": float(target_xyz[0]),
        "target_y": float(target_xyz[1]),
        "target_z": float(target_xyz[2]),
        "true_x": truth.x,
        "true_y": truth.y,
        "true_z": truth.z,
        "est_x": estimate.x,
        "est_y": estimate.y,
        "est_z": estimate.z,
        "est_psi_deg": debug.get("est_psi_deg", math.degrees(estimate.psi)),
        "est_theta_deg": debug.get("est_theta_deg", math.degrees(estimate.theta)),
        "est_vx": debug.get("est_vx", estimate.vx),
        "est_vy": debug.get("est_vy", estimate.vy),
        "est_vz": debug.get("est_vz", estimate.vz),
        "u": truth.u,
        "psi_deg": math.degrees(truth.psi),
        "r_deg_s": math.degrees(truth.r),
        "w": truth.w,
        "theta_deg": math.degrees(truth.theta),
        "q_deg_s": math.degrees(truth.q),
        "delta_cmd_deg": math.degrees(command.delta_cmd),
        "delta_act_deg": math.degrees(truth.delta_act),
        "elev_cmd_deg": math.degrees(command.elev_cmd),
        "elev_act_deg": math.degrees(truth.elev_act),
        "thrust": command.thrust,
        "buoyancy_cmd": command.buoyancy_cmd,
        "distance_traveled_m": truth.distance_traveled_m,
        "waypoint_distance_true_m": true_xy,
        "waypoint_distance_est_m": est_xy,
        "nav_mode": debug.get("nav_mode", ""),
        "nav_status": debug.get("nav_status", ""),
        "nav_health": debug.get("nav_health", ""),
        "nav_use_current_state": bool(debug.get("nav_use_current_state", False)),
        "nav_pos_std_m": debug.get("nav_pos_std_m", ""),
        "nav_vel_std_mps": debug.get("nav_vel_std_mps", ""),
        "nav_yaw_std_deg": debug.get("nav_yaw_std_deg", ""),
        "imu_used": bool(debug.get("imu_used", False)),
        "depth_used": bool(debug.get("depth_used", False)),
        "depth_accepted": bool(debug.get("depth_accepted", False)),
        "depth_innovation_m": debug.get("depth_innovation_m", ""),
        "compass_used": bool(debug.get("compass_used", False)),
        "compass_accepted": bool(debug.get("compass_accepted", False)),
        "compass_innovation_deg": debug.get("compass_innovation_deg", ""),
        "dvl_used": bool(debug.get("dvl_used", False)),
        "dvl_accepted": bool(debug.get("dvl_accepted", False)),
        "dvl_bottom_lock": bool(debug.get("dvl_bottom_lock", False)),
        "dvl_altitude_m": debug.get("dvl_altitude_m", ""),
        "dvl_innovation_norm_mps": debug.get("dvl_innovation_norm_mps", ""),
        "dvl_vel_blend_gain": debug.get("dvl_vel_blend_gain", ""),
        "dvl_pos_blend_gain": debug.get("dvl_pos_blend_gain", ""),
        "gps_used": bool(debug.get("gps_used", False)),
        "gps_accepted": bool(debug.get("gps_accepted", False)),
        "gps_innovation_m": debug.get("gps_innovation_m", ""),
        "est_current_x": debug.get("est_current_x", ""),
        "est_current_y": debug.get("est_current_y", ""),
        "est_current_z": debug.get("est_current_z", ""),
        "bathy_enabled": bool(debug.get("bathy_enabled", False)),
        "bathy_valid_beams": debug.get("bathy_valid_beams", ""),
        "bathy_used": bool(debug.get("bathy_used", False)),
        "bathy_accepted": bool(debug.get("bathy_accepted", False)),
        "bathy_lock": bool(debug.get("bathy_lock", False)),
        "bathy_best_score": debug.get("bathy_best_score", ""),
        "bathy_prior_score": debug.get("bathy_prior_score", ""),
        "bathy_score_improvement": debug.get("bathy_score_improvement", ""),
        "bathy_gradient_norm": debug.get("bathy_gradient_norm", ""),
        "bathy_best_x": debug.get("bathy_best_x", ""),
        "bathy_best_y": debug.get("bathy_best_y", ""),
        "bathy_corrected_x": debug.get("bathy_corrected_x", ""),
        "bathy_corrected_y": debug.get("bathy_corrected_y", ""),
        "bathy_center_elev_m": debug.get("bathy_center_elev_m", ""),
        "bathy_status": debug.get("bathy_status", ""),
        "bathy_innovation_xy_m": debug.get("bathy_innovation_xy_m", ""),
        "correction_x": debug.get("correction_x", ""),
        "correction_y": debug.get("correction_y", ""),
        "attached": attached,
        "parent_vehicle": parent_vehicle,
        "notes": notes,
    }


def make_event_log_row(
    timestamp_s: float,
    vehicle_id: str,
    event_type: str,
    route_id: str = "",
    notes: str = "",
) -> Dict[str, Any]:
    row = {key: "" for key in LOG_COLUMNS}
    row["timestamp_s"] = round(float(timestamp_s), 3)
    row["vehicle_id"] = vehicle_id
    row["row_type"] = "event"
    row["event_type"] = event_type
    row["route_id"] = route_id
    row["notes"] = notes
    return row


def write_vehicle_csv(rows: Iterable[Dict[str, Any]], output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    extra_keys: List[str] = []
    seen = set(LOG_COLUMNS)
    for row in row_list:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                extra_keys.append(key)
    fieldnames = list(LOG_COLUMNS) + extra_keys
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in row_list:
            writer.writerow(row)


# ============================================================
# Visualization helpers
# ============================================================


def _heading_segments(x: Sequence[float], y: Sequence[float], psi: Sequence[float], stride: int, length: float) -> Tuple[List[float], List[float], List[float], List[float]]:
    xs, ys, us, vs = [], [], [], []
    for idx in range(0, len(x), max(1, int(stride))):
        xs.append(float(x[idx]))
        ys.append(float(y[idx]))
        us.append(length * math.cos(float(psi[idx])))
        vs.append(length * math.sin(float(psi[idx])))
    return xs, ys, us, vs


def plot_mission_2d(
    planned_xyz: Dict[str, List[Tuple[float, float, float]]],
    truth_histories: Dict[str, Dict[str, Sequence[float]]],
    est_histories: Optional[Dict[str, Dict[str, Sequence[float]]]] = None,
    show_heading: bool = True,
    heading_stride: int = 20,
) -> Tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(9, 9))
    for name, plan in planned_xyz.items():
        if plan:
            ax.plot([p[0] for p in plan], [p[1] for p in plan], "--", alpha=0.7, label=f"{name} planned")

    for name, hist in truth_histories.items():
        ax.plot(hist["x"], hist["y"], linewidth=2, label=f"{name} truth")
        if show_heading and "psi" in hist and len(hist["x"]) > 0:
            xs, ys, us, vs = _heading_segments(hist["x"], hist["y"], hist["psi"], heading_stride, 5.0)
            ax.quiver(xs, ys, us, vs, angles="xy", scale_units="xy", scale=1.0, width=0.0025, alpha=0.5)

    if est_histories:
        for name, hist in est_histories.items():
            ax.plot(hist["x"], hist["y"], ":", alpha=0.9, label=f"{name} estimate")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Mission Profile: Planned vs Actual")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(loc="best")
    return fig, ax


def animate_mission_2d(
    planned_xyz: Dict[str, List[Tuple[float, float, float]]],
    truth_histories: Dict[str, Dict[str, Sequence[float]]],
    est_histories: Optional[Dict[str, Dict[str, Sequence[float]]]] = None,
    dt: float = 0.1,
    playback_speed: float = 2.0,
) -> FuncAnimation:
    all_x = []
    all_y = []
    for plan in planned_xyz.values():
        all_x.extend([p[0] for p in plan])
        all_y.extend([p[1] for p in plan])
    for hist in truth_histories.values():
        all_x.extend(hist["x"])
        all_y.extend(hist["y"])
    if est_histories:
        for hist in est_histories.values():
            all_x.extend(hist["x"])
            all_y.extend(hist["y"])

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(min(all_x) - 20.0, max(all_x) + 20.0)
    ax.set_ylim(min(all_y) - 20.0, max(all_y) + 20.0)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Mission Playback 2D")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    planned_lines, trails, points = {}, {}, {}
    est_trails, est_points = {}, {}
    for name, plan in planned_xyz.items():
        if plan:
            planned_lines[name] = ax.plot([p[0] for p in plan], [p[1] for p in plan], "--", alpha=0.6)[0]
        trails[name] = ax.plot([], [], linewidth=2, label=f"{name} truth")[0]
        points[name] = ax.plot([], [], marker="o", linestyle="None")[0]
        if est_histories and name in est_histories:
            est_trails[name] = ax.plot([], [], ":", alpha=0.8, label=f"{name} estimate")[0]
            est_points[name] = ax.plot([], [], marker="x", linestyle="None")[0]
    ax.legend(loc="best")

    n_frames = max(len(hist["x"]) for hist in truth_histories.values())
    interval_ms = max(20, int((1000.0 * dt) / max(playback_speed, 1e-6)))

    def update(frame: int):
        artists = []
        for name, hist in truth_histories.items():
            idx = min(frame, len(hist["x"]) - 1)
            trails[name].set_data(hist["x"][: idx + 1], hist["y"][: idx + 1])
            points[name].set_data([hist["x"][idx]], [hist["y"][idx]])
            artists.extend([trails[name], points[name]])
            if est_histories and name in est_histories:
                est = est_histories[name]
                idxe = min(frame, len(est["x"]) - 1)
                est_trails[name].set_data(est["x"][: idxe + 1], est["y"][: idxe + 1])
                est_points[name].set_data([est["x"][idxe]], [est["y"][idxe]])
                artists.extend([est_trails[name], est_points[name]])
        return artists

    return FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=False, repeat=False)


def plot_mission_3d(
    planned_xyz: Dict[str, List[Tuple[float, float, float]]],
    truth_histories: Dict[str, Dict[str, Sequence[float]]],
    est_histories: Optional[Dict[str, Dict[str, Sequence[float]]]] = None,
) -> Tuple[Any, Any]:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for name, plan in planned_xyz.items():
        if plan:
            ax.plot([p[0] for p in plan], [p[1] for p in plan], [p[2] for p in plan], "--", alpha=0.7, label=f"{name} planned")
    for name, hist in truth_histories.items():
        ax.plot(hist["x"], hist["y"], hist["z"], linewidth=2, label=f"{name} truth")
    if est_histories:
        for name, hist in est_histories.items():
            ax.plot(hist["x"], hist["y"], hist["z"], ":", alpha=0.8, label=f"{name} estimate")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m, local NED]")
    ax.set_title("Mission Profile: Isometric 3D")
    ax.view_init(elev=25, azim=-60)
    ax.legend(loc="best")
    return fig, ax


def animate_mission_3d(
    planned_xyz: Dict[str, List[Tuple[float, float, float]]],
    truth_histories: Dict[str, Dict[str, Sequence[float]]],
    est_histories: Optional[Dict[str, Dict[str, Sequence[float]]]] = None,
    dt: float = 0.1,
    playback_speed: float = 2.0,
) -> FuncAnimation:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for name, plan in planned_xyz.items():
        if plan:
            ax.plot([p[0] for p in plan], [p[1] for p in plan], [p[2] for p in plan], "--", alpha=0.5)

    trails, points = {}, {}
    est_trails, est_points = {}, {}
    for name in truth_histories:
        trails[name], = ax.plot([], [], [], linewidth=2, label=f"{name} truth")
        points[name], = ax.plot([], [], [], marker="o", linestyle="None")
        if est_histories and name in est_histories:
            est_trails[name], = ax.plot([], [], [], ":", alpha=0.8, label=f"{name} estimate")
            est_points[name], = ax.plot([], [], [], marker="x", linestyle="None")

    n_frames = max(len(hist["x"]) for hist in truth_histories.values())
    interval_ms = max(20, int((1000.0 * dt) / max(playback_speed, 1e-6)))
    ax.legend(loc="best")

    def update(frame: int):
        artists = []
        for name, hist in truth_histories.items():
            idx = min(frame, len(hist["x"]) - 1)
            trails[name].set_data(hist["x"][: idx + 1], hist["y"][: idx + 1])
            trails[name].set_3d_properties(hist["z"][: idx + 1])
            points[name].set_data([hist["x"][idx]], [hist["y"][idx]])
            points[name].set_3d_properties([hist["z"][idx]])
            artists.extend([trails[name], points[name]])
            if est_histories and name in est_histories:
                est = est_histories[name]
                idxe = min(frame, len(est["x"]) - 1)
                est_trails[name].set_data(est["x"][: idxe + 1], est["y"][: idxe + 1])
                est_trails[name].set_3d_properties(est["z"][: idxe + 1])
                est_points[name].set_data([est["x"][idxe]], [est["y"][idxe]])
                est_points[name].set_3d_properties([est["z"][idxe]])
                artists.extend([est_trails[name], est_points[name]])
        return artists

    return FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=False, repeat=False)




# ============================================================
# History helpers for later integration
# ============================================================


def init_history() -> Dict[str, List[float]]:
    return {key: [] for key in ["t", "x", "y", "z", "u", "psi", "r", "w", "theta", "q", "delta_act", "elev_act"]}


def append_truth_history(history: Dict[str, List[float]], state: TruthState) -> None:
    history["t"].append(state.t)
    history["x"].append(state.x)
    history["y"].append(state.y)
    history["z"].append(state.z)
    history["u"].append(state.u)
    history["psi"].append(state.psi)
    history["r"].append(state.r)
    history["w"].append(state.w)
    history["theta"].append(state.theta)
    history["q"].append(state.q)
    history["delta_act"].append(state.delta_act)
    history["elev_act"].append(state.elev_act)


def init_estimate_history() -> Dict[str, List[float]]:
    return {key: [] for key in ["x", "y", "z", "psi", "vx", "vy", "vz"]}


def append_estimate_history(history: Dict[str, List[float]], state: EstimateState) -> None:
    history["x"].append(state.x)
    history["y"].append(state.y)
    history["z"].append(state.z)
    history["psi"].append(state.psi)
    history["vx"].append(state.vx)
    history["vy"].append(state.vy)
    history["vz"].append(state.vz)
