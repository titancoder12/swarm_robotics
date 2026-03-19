from __future__ import annotations

import numpy as np


class RuleBasedSwarmPolicy:
    """A lightweight baseline policy that consumes the standard 23-dim observation."""

    def __init__(self, cfg, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._action_lookup = {
            (-1, -1): 0,
            (-1, 0): 1,
            (-1, 1): 2,
            (0, -1): 3,
            (0, 0): 4,
            (0, 1): 5,
            (1, -1): 6,
            (1, 0): 7,
            (1, 1): 8,
        }

    def act(self, obs: np.ndarray) -> int:
        parts = self._split_obs(obs)
        lidar = parts["lidar"]
        target_vec = parts["target_vec"]
        nest_vec = parts["nest_vec"]
        food_presence = parts["food_presence"]
        carrying = parts["carrying"]
        pheromone = parts["pheromone"]

        if carrying > 0.5 and np.linalg.norm(nest_vec) > 0.05:
            return self._steer_toward(nest_vec, lidar)
        if food_presence > 0.5 or np.linalg.norm(target_vec) > 0.10:
            return self._steer_toward(target_vec, lidar)
        if float(np.max(pheromone)) > 0.05:
            return self._follow_pheromone(pheromone, lidar)
        return self._explore(lidar)

    def _split_obs(self, obs: np.ndarray) -> dict[str, np.ndarray | float]:
        idx = 0
        lidar = obs[idx : idx + self.cfg.lidar_rays]
        idx += self.cfg.lidar_rays
        target_vec = obs[idx : idx + 2]
        idx += 2
        nest_vec = np.zeros(2, dtype=np.float32)
        if self.cfg.obs_include_nest_direction:
            nest_vec = obs[idx : idx + 2]
            idx += 2
        idx += 2  # nearest agent vector
        idx += 2  # heading
        idx += 1  # speed
        food_presence = 0.0
        if self.cfg.obs_include_food_presence:
            food_presence = float(obs[idx])
            idx += 1
        carrying = 0.0
        if self.cfg.obs_include_carrying:
            carrying = float(obs[idx])
            idx += 1
        pheromone = obs[idx : idx + self.cfg.pheromone_samples]
        return {
            "lidar": lidar,
            "target_vec": target_vec,
            "nest_vec": nest_vec,
            "food_presence": food_presence,
            "carrying": carrying,
            "pheromone": pheromone,
        }

    def _steer_toward(self, vec: np.ndarray, lidar: np.ndarray) -> int:
        turn = 0
        lateral = float(vec[1])
        if lateral > 0.12:
            turn = 1
        elif lateral < -0.12:
            turn = -1

        throttle = 1
        if self._front_blocked(lidar):
            turn = self._clearer_turn(lidar)
            throttle = 0
        return self._action_lookup[(throttle, turn)]

    def _follow_pheromone(self, pheromone: np.ndarray, lidar: np.ndarray) -> int:
        if self._front_blocked(lidar):
            return self._action_lookup[(0, self._clearer_turn(lidar))]
        near = float(np.mean(pheromone[: max(1, len(pheromone) // 2)]))
        far = float(np.mean(pheromone[max(1, len(pheromone) // 2) :]))
        if far >= near:
            return self._action_lookup[(1, 0)]
        turn = int(self.rng.choice([-1, 1]))
        return self._action_lookup[(0, turn)]

    def _explore(self, lidar: np.ndarray) -> int:
        if self._front_blocked(lidar):
            return self._action_lookup[(0, self._clearer_turn(lidar))]
        turn = 0
        if self.rng.random() < 0.2:
            turn = int(self.rng.choice([-1, 1]))
        return self._action_lookup[(1, turn)]

    def _front_blocked(self, lidar: np.ndarray) -> bool:
        center = len(lidar) // 2
        window = lidar[max(0, center - 1) : min(len(lidar), center + 2)]
        return float(np.min(window)) < 0.18

    def _clearer_turn(self, lidar: np.ndarray) -> int:
        mid = len(lidar) // 2
        left_clearance = float(np.mean(lidar[:mid])) if mid > 0 else 0.0
        right_clearance = float(np.mean(lidar[mid + 1 :])) if mid + 1 < len(lidar) else 0.0
        return 1 if left_clearance >= right_clearance else -1
