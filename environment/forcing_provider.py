from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from guidance_and_control.guidance_core import EnvForcing


@dataclass
class JsonScenarioForcingProvider:
    sim_cfg: Dict[str, Any]
    sim_cfg_source_path: str | None = None

    def __post_init__(self) -> None:
        forcing_cfg = self.sim_cfg.get("environment_forcing", {})
        self.mode = str(forcing_cfg.get("mode", "none")).lower()
        self.scenario = str(forcing_cfg.get("scenario", "minus_1sd"))
        self.current_only = bool(forcing_cfg.get("current_only", True))
        self.interpolation = str(forcing_cfg.get("interpolation", "hold")).lower()
        self.time_reference = dict(forcing_cfg.get("time_reference", {}))

        source = forcing_cfg.get("source")
        self.payload: Dict[str, Any] = {}
        self.surface_xy = np.zeros(2, dtype=float)
        self.underwater_xy = np.zeros(2, dtype=float)
        self.current_z = 0.0

        if self.mode != "json_scenario" or not source:
            return

        source_path = Path(source)
        if not source_path.is_absolute():
            if self.sim_cfg_source_path:
                cfg_dir = Path(self.sim_cfg_source_path).resolve().parent
                source_path = (cfg_dir / source_path).resolve()
            else:
                source_path = source_path.resolve()

        with source_path.open("r", encoding="utf-8") as handle:
            self.payload = json.load(handle)

        env = self.payload["env_blocks"][self.scenario]["environment"]

        self.surface_xy = np.asarray(env.get("surface_current_xy", [0.0, 0.0]), dtype=float)
        self.underwater_xy = np.asarray(env.get("underwater_current_xy", [0.0, 0.0]), dtype=float)
        self.current_z = float(env.get("underwater_current_z", 0.0))

    def forcing_at_time(self, t_s: float, vehicle_mode: str) -> EnvForcing:
        # For now this is static even though the interface is time-aware.
        # Later you can swap this class for a timestepped provider.
        if vehicle_mode == "surface":
            return EnvForcing(current_xy=self.surface_xy.copy(), current_z=0.0)

        return EnvForcing(current_xy=self.underwater_xy.copy(), current_z=self.current_z)