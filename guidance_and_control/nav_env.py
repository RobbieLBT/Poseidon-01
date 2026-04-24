from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from guidance_and_control.guidance_core import EnvForcing, EstimateState, TruthState
from guidance_and_control.sensors.bathymetry_sensor import (
    BathymetryGrid,
    ConeBeamSensorConfig,
    TerrainAidConfig,
    _simulate_cone_beam_measurement_impl,
    _terrain_aided_position_update_impl,
)


DEFAULT_GRAVITY_MPS2 = 9.80665
DEFAULT_BOTTOM_LOCK_ALTITUDE_RANGE_M = (0.5, 250.0)


@dataclass
class IMUSensorConfig:
    enabled: bool = True
    gyro_noise_std_rad_s: np.ndarray = field(default_factory=lambda: np.full(3, math.radians(0.08), dtype=float))
    accel_noise_std_mps2: np.ndarray = field(default_factory=lambda: np.full(3, 0.04, dtype=float))
    gyro_bias_init_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    accel_bias_init_mps2: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    gyro_bias_rw_rad_s_sqrt_s: np.ndarray = field(default_factory=lambda: np.full(3, math.radians(0.005), dtype=float))
    accel_bias_rw_mps2_sqrt_s: np.ndarray = field(default_factory=lambda: np.full(3, 0.002, dtype=float))


@dataclass
class DepthSensorConfig:
    enabled: bool = True
    update_rate_hz: float = 4.0
    sigma_m: float = 0.3
    bias_m: float = 0.0
    dropout_prob: float = 0.0


@dataclass
class CompassSensorConfig:
    enabled: bool = True
    update_rate_hz: float = 4.0
    sigma_rad: float = math.radians(2.0)
    bias_rad: float = 0.0
    dropout_prob: float = 0.0


@dataclass
class DVLSensorConfig:
    enabled: bool = True
    mode: str = "bottom_lock"
    update_rate_hz: float = 5.0
    sigma_mps: np.ndarray = field(default_factory=lambda: np.full(3, 0.05, dtype=float))
    bias_mps: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    dropout_prob: float = 0.0
    min_altitude_m: float = DEFAULT_BOTTOM_LOCK_ALTITUDE_RANGE_M[0]
    max_altitude_m: float = DEFAULT_BOTTOM_LOCK_ALTITUDE_RANGE_M[1]
    assume_lock_without_map: bool = True


@dataclass
class GPSSensorConfig:
    enabled: bool = False
    update_rate_hz: float = 1.0
    sigma_h_m: float = 2.0
    sigma_v_m: float = 3.0
    surfaced_only: bool = True
    surface_depth_threshold_m: float = 0.75
    dropout_prob: float = 0.0


@dataclass
class ESKFConfig:
    enabled: bool = True
    gravity_mps2: float = DEFAULT_GRAVITY_MPS2
    use_current_bias_state: bool = False
    use_disturbance_state: bool = False
    current_bias_tau_s: float = 300.0
    disturbance_tau_s: float = 60.0
    position_init_sigma_m: float = 1.0
    velocity_init_sigma_mps: float = 0.5
    roll_init_sigma_deg: float = 5.0
    pitch_init_sigma_deg: float = 5.0
    yaw_init_sigma_deg: float = 8.0
    gyro_bias_init_sigma_rad_s: float = math.radians(0.2)
    accel_bias_init_sigma_mps2: float = 0.15
    current_bias_init_sigma_mps: float = 0.15
    disturbance_init_sigma_mps2: float = 0.05
    gyro_noise_std_rad_s: float = math.radians(0.10)
    accel_noise_std_mps2: float = 0.08
    gyro_bias_rw_rad_s_sqrt_s: float = math.radians(0.01)
    accel_bias_rw_mps2_sqrt_s: float = 0.01
    current_bias_rw_mps_sqrt_s: float = 0.02
    disturbance_rw_mps2_sqrt_s: float = 0.02
    depth_gate_chi2: float = 9.0
    compass_gate_chi2: float = 9.0
    dvl_gate_chi2: float = 16.0
    gps_gate_chi2: float = 25.0
    bathy_gate_chi2: float = 16.0
    bathy_position_sigma_base_m: float = 2.0
    bathy_position_sigma_per_score_m: float = 0.20
    bathy_position_sigma_max_m: float = 12.0


@dataclass
class NominalNavState:
    p_ned_m: np.ndarray
    v_ned_mps: np.ndarray
    q_nb: np.ndarray
    gyro_bias_rad_s: np.ndarray
    accel_bias_mps2: np.ndarray
    current_bias_ned_mps: np.ndarray
    disturbance_ned_mps2: np.ndarray

    def copy(self) -> "NominalNavState":
        return NominalNavState(
            p_ned_m=self.p_ned_m.copy(),
            v_ned_mps=self.v_ned_mps.copy(),
            q_nb=self.q_nb.copy(),
            gyro_bias_rad_s=self.gyro_bias_rad_s.copy(),
            accel_bias_mps2=self.accel_bias_mps2.copy(),
            current_bias_ned_mps=self.current_bias_ned_mps.copy(),
            disturbance_ned_mps2=self.disturbance_ned_mps2.copy(),
        )


@dataclass
class NavStepResult:
    estimate: EstimateState
    debug: Dict[str, Any]

    def as_estimate_state(self) -> EstimateState:
        return self.estimate


class AUVNavigationStack:
    """ESKF-style AUV navigation stack.

    Internal conventions:
    - navigation frame: NED (+down)
    - body frame: FRD
    - quaternion q_nb rotates body vectors into navigation

    The surrounding simulator still uses the existing mission convention where
    submerged z values are negative. This class converts at the boundary so the
    estimator remains NED internally while the rest of the runtime stays compatible.
    """

    def __init__(
        self,
        runtime_cfg: Dict[str, Any],
        initial_truth: TruthState,
        initial_forcing: Optional[EnvForcing] = None,
        rng: Optional[np.random.Generator] = None,
        sim_cfg_source_path: Optional[str] = None,
    ) -> None:
        self.runtime_cfg = runtime_cfg
        self.est_cfg = runtime_cfg.get("estimation", {})
        self.drift_cfg = runtime_cfg.get("drift", {})
        self.sensor_root = runtime_cfg.get("sensors", {})
        self.sim_cfg_source_path = sim_cfg_source_path
        self.rng = rng if rng is not None else np.random.default_rng(7)

        self.eskf_cfg = self._parse_eskf_cfg(self.est_cfg)
        self.imu_cfg = self._parse_imu_cfg(self.sensor_root.get("imu", {}))
        self.depth_cfg = self._parse_depth_cfg(self.sensor_root.get("depth", {}))
        self.compass_cfg = self._parse_compass_cfg(self.sensor_root.get("compass", {}))
        self.dvl_cfg = self._parse_dvl_cfg(self.sensor_root.get("dvl", {}))
        self.gps_cfg = self._parse_gps_cfg(self.sensor_root.get("gps", {}))
        self.bathy_sensor_cfg = dict(self.sensor_root.get("bathymetry_cone", {}))
        self.bathy_grid = self._load_bathymetry_grid(self.bathy_sensor_cfg)

        self._build_error_state_layout()
        initial_forcing = initial_forcing if initial_forcing is not None else EnvForcing()
        self.state = self._state_from_truth(initial_truth, initial_forcing, apply_drift_init=True)
        self.P = self._initial_covariance()

        self.true_gyro_bias_rad_s = self.imu_cfg.gyro_bias_init_rad_s.copy()
        self.true_accel_bias_mps2 = self.imu_cfg.accel_bias_init_mps2.copy()

        self.next_depth_time_s = 0.0
        self.next_compass_time_s = 0.0
        self.next_dvl_time_s = 0.0
        self.next_gps_time_s = 0.0
        self.next_bathy_time_s = 0.0

        self.last_debug = self._base_debug()
        self.last_debug.update(self._estimate_debug_fields())
        self._last_truth = initial_truth
        self._sim_time_s = 0.0

    def reset_from_truth(
        self,
        truth: TruthState,
        forcing: EnvForcing,
        *,
        apply_drift_init: bool = False,
        preserve_timing: bool = True,
    ) -> EstimateState:
        self.state = self._state_from_truth(truth, forcing, apply_drift_init=apply_drift_init)
        self.P = self._initial_covariance()
        self.last_debug = self._base_debug()
        self.last_debug.update(self._estimate_debug_fields())
        if not preserve_timing:
            self.next_depth_time_s = 0.0
            self.next_compass_time_s = 0.0
            self.next_dvl_time_s = 0.0
            self.next_gps_time_s = 0.0
            self.next_bathy_time_s = 0.0
        self._last_truth = truth
        self._sim_time_s = 0.0
        return self.as_estimate_state()

    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Compatibility wrapper.

        Supported calling conventions:
        - step(prev_truth, next_truth, forcing, dt, sim_time_s) -> (EstimateState, debug)
        - step(next_truth, forcing, dt) -> NavStepResult
        """
        if len(args) == 5:
            prev_truth, next_truth, forcing, dt, sim_time_s = args
            return self._step_core(prev_truth, next_truth, forcing, dt, sim_time_s)
        if len(args) == 3 and not kwargs:
            next_truth, forcing, dt = args
            prev_truth = getattr(self, '_last_truth', next_truth)
            sim_time_s = float(getattr(self, '_sim_time_s', 0.0))
            estimate, debug = self._step_core(prev_truth, next_truth, forcing, dt, sim_time_s)
            self._last_truth = next_truth
            self._sim_time_s = sim_time_s + float(dt)
            return NavStepResult(estimate=estimate, debug=debug)
        raise TypeError('AUVNavigationStack.step expects either (prev_truth, next_truth, forcing, dt, sim_time_s) or (next_truth, forcing, dt)')

    # keep the rest of your existing nav_env.py exactly as-is,
    # but rename the original step(...) implementation body to _step_core(...)
    # with the signature:
    # def _step_core(self, prev_truth, next_truth, forcing, dt, sim_time_s):
    #     ... existing estimator logic ...
    # and keep the existing _load_bathymetry_grid method, which already resolves
    # paths relative to sim_config.yaml correctly.