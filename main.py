from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import random
from typing import Dict, List, Tuple

import numpy as np


GridPoint = Tuple[int, int]


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw], dtype=float)

    @staticmethod
    def from_array(arr: np.ndarray) -> "Pose2D":
        return Pose2D(float(arr[0]), float(arr[1]), float(arr[2]))


def gaussian_fusion(mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    i1 = np.linalg.inv(cov1)
    i2 = np.linalg.inv(cov2)
    cov_f = np.linalg.inv(i1 + i2)
    mu_f = cov_f @ (i1 @ mu1 + i2 @ mu2)
    return mu_f, cov_f


class SemanticGridMap:
    """Stores per-cell class logits and cleaned mask."""

    classes = ["panel", "frame", "gap", "obstacle", "background", "cleaned"]

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.logits = np.zeros((height, width, len(self.classes)), dtype=float)
        self.cleaned = np.zeros((height, width), dtype=bool)

    def update_cell(self, x: int, y: int, class_id: int, confidence: float) -> None:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return
        confidence = float(np.clip(confidence, 1e-3, 1 - 1e-3))
        log_odds = math.log(confidence / (1.0 - confidence))
        self.logits[y, x, class_id] += log_odds

    def class_prob(self, x: int, y: int) -> np.ndarray:
        l = self.logits[y, x]
        expv = np.exp(l - np.max(l))
        return expv / np.sum(expv)

    def mark_cleaned(self, x: int, y: int) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cleaned[y, x] = True
            self.update_cell(x, y, self.classes.index("cleaned"), 0.8)

    def cleaned_ratio(self, panel_mask: np.ndarray) -> float:
        target = np.sum(panel_mask)
        if target == 0:
            return 0.0
        covered = np.sum(np.logical_and(panel_mask, self.cleaned))
        return float(covered / target)


class PVEnvironment:
    def __init__(self, width: int = 64, height: int = 36) -> None:
        self.width = width
        self.height = height
        self.panel_mask = np.zeros((height, width), dtype=bool)
        self.obstacle_mask = np.zeros((height, width), dtype=bool)
        self._build_scene()

    def _build_scene(self) -> None:
        # Panel rows
        for y in range(4, self.height - 4):
            if y % 4 in (1, 2):
                self.panel_mask[y, 3 : self.width - 3] = True
        # Gaps and obstacles
        for x in range(12, self.width - 12, 14):
            self.panel_mask[:, x : x + 1] = False
        for _ in range(18):
            ox = random.randint(6, self.width - 7)
            oy = random.randint(6, self.height - 7)
            self.obstacle_mask[oy - 1 : oy + 2, ox - 1 : ox + 2] = True
            self.panel_mask[oy - 1 : oy + 2, ox - 1 : ox + 2] = False

    def in_bounds(self, p: GridPoint) -> bool:
        x, y = p
        return 0 <= x < self.width and 0 <= y < self.height

    def traversable(self, p: GridPoint) -> bool:
        x, y = p
        return self.in_bounds(p) and self.panel_mask[y, x] and not self.obstacle_mask[y, x]

    def boundary_distance(self, p: GridPoint) -> float:
        x, y = p
        # Approximate boundary as nearest non-panel cell distance
        best = 1e9
        for yy in range(max(0, y - 5), min(self.height, y + 6)):
            for xx in range(max(0, x - 5), min(self.width, x + 6)):
                if not self.panel_mask[yy, xx]:
                    d = math.hypot(xx - x, yy - y)
                    best = min(best, d)
        return best if best < 1e8 else 6.0


class UncertaintyAwarePlanner:
    def __init__(self, env: PVEnvironment) -> None:
        self.env = env

    def _neighbors(self, p: GridPoint) -> List[GridPoint]:
        x, y = p
        cand = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [q for q in cand if self.env.traversable(q)]

    def _heuristic(self, a: GridPoint, b: GridPoint) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan(
        self,
        start: GridPoint,
        goal: GridPoint,
        uncertainty_map: np.ndarray,
        cleaned_map: np.ndarray,
        alpha_len: float = 1.0,
        beta_risk: float = 2.4,
        gamma_unc: float = 1.4,
        delta_repeat: float = 1.2,
    ) -> List[GridPoint]:
        open_heap: List[Tuple[float, GridPoint]] = []
        heapq.heappush(open_heap, (0.0, start))
        g_cost: Dict[GridPoint, float] = {start: 0.0}
        parent: Dict[GridPoint, GridPoint] = {}

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == goal:
                break

            for nxt in self._neighbors(cur):
                bx = self.env.boundary_distance(nxt)
                risk = 1.0 / (bx + 1e-3)
                unc = float(uncertainty_map[nxt[1], nxt[0]])
                repeat = 1.0 if cleaned_map[nxt[1], nxt[0]] else 0.0
                step = alpha_len + beta_risk * risk + gamma_unc * unc + delta_repeat * repeat
                tentative = g_cost[cur] + step
                if tentative < g_cost.get(nxt, 1e18):
                    g_cost[nxt] = tentative
                    parent[nxt] = cur
                    f = tentative + self._heuristic(nxt, goal)
                    heapq.heappush(open_heap, (f, nxt))

        if goal not in parent and goal != start:
            return [start]
        path = [goal]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
        return path


class MultiModalSLAMDemo:
    def __init__(self, env: PVEnvironment) -> None:
        self.env = env
        self.semantic_map = SemanticGridMap(env.width, env.height)
        self.pose = Pose2D(5.0, 5.0, 0.0)
        self.cov = np.diag([0.3, 0.3, 0.1])
        self.planner = UncertaintyAwarePlanner(env)
        self.uncertainty_grid = np.ones((env.height, env.width), dtype=float) * 0.5

    def _simulate_sensor_estimates(self, target: Pose2D) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # VIO estimate (good short-term, light sensitive)
        mu_vio = target.as_array() + np.random.normal(0, [0.25, 0.25, 0.06], size=3)
        cov_vio = np.diag([0.25, 0.25, 0.08])

        # LIO estimate (good geometry, may degenerate on repeated planes)
        deg = random.random() < 0.15
        sigma_xy = 0.18 if not deg else 0.45
        mu_lio = target.as_array() + np.random.normal(0, [sigma_xy, sigma_xy, 0.04], size=3)
        cov_lio = np.diag([sigma_xy, sigma_xy, 0.06])

        return mu_vio, cov_vio, mu_lio, cov_lio

    def _semantic_quality(self, p: GridPoint) -> float:
        x, y = p
        glare = 0.25 if (x + y) % 11 == 0 else 0.0
        obstacle_near = 0.35 if self.env.boundary_distance(p) < 1.6 else 0.0
        q = 1.0 - glare - obstacle_near
        return float(np.clip(q, 0.05, 0.98))

    def step(self, goal: GridPoint) -> List[GridPoint]:
        target = Pose2D(goal[0], goal[1], 0.0)
        mu_vio, cov_vio, mu_lio, cov_lio = self._simulate_sensor_estimates(target)
        mu_f, cov_f = gaussian_fusion(mu_vio, cov_vio, mu_lio, cov_lio)

        # Semantic confidence affects uncertainty score U_t
        qv = 1.0 - min(1.0, np.trace(cov_vio) / 2.0)
        ql = 1.0 - min(1.0, np.trace(cov_lio) / 2.0)
        qs = self._semantic_quality(goal)
        u_t = float(np.trace(cov_f) + 0.8 * (1 - qv) + 0.8 * (1 - ql) + 0.6 * (1 - qs))

        self.pose = Pose2D.from_array(mu_f)
        self.cov = cov_f

        gx, gy = goal
        self.uncertainty_grid[gy, gx] = 0.6 * self.uncertainty_grid[gy, gx] + 0.4 * u_t
        cls = SemanticGridMap.classes.index("panel" if self.env.panel_mask[gy, gx] else "background")
        self.semantic_map.update_cell(gx, gy, cls, qs)

        start = (int(round(self.pose.x)), int(round(self.pose.y)))
        if not self.env.traversable(start):
            start = goal
        path = self.planner.plan(start, goal, self.uncertainty_grid, self.semantic_map.cleaned)
        for px, py in path:
            self.semantic_map.mark_cleaned(px, py)
        return path


def generate_cleaning_targets(env: PVEnvironment, limit: int = 80) -> List[GridPoint]:
    targets: List[GridPoint] = []
    for y in range(env.height):
        xs = range(env.width) if y % 2 == 0 else range(env.width - 1, -1, -1)
        for x in xs:
            if env.traversable((x, y)):
                targets.append((x, y))
                if len(targets) >= limit:
                    return targets
    return targets


def main() -> None:
    np.random.seed(7)
    random.seed(7)

    env = PVEnvironment(64, 36)
    slam = MultiModalSLAMDemo(env)
    targets = generate_cleaning_targets(env, limit=120)

    total_len = 0
    for t in targets:
        path = slam.step(t)
        total_len += max(0, len(path) - 1)

    coverage = slam.semantic_map.cleaned_ratio(env.panel_mask)
    mean_u = float(np.mean(slam.uncertainty_grid[env.panel_mask]))
    ate_proxy = float(np.trace(slam.cov) ** 0.5)
    safe_margin = np.min(
        [env.boundary_distance((x, y)) for y in range(env.height) for x in range(env.width) if slam.semantic_map.cleaned[y, x]]
        or [0.0]
    )

    print("=== Multi-Modal Semantic SLAM + Uncertainty-Aware Navigation Demo ===")
    print(f"Map size: {env.width} x {env.height}")
    print(f"Executed cleaning targets: {len(targets)}")
    print(f"Approx path length: {total_len}")
    print(f"Coverage ratio: {coverage * 100:.2f}%")
    print(f"Mean uncertainty U: {mean_u:.4f}")
    print(f"ATE proxy (sqrt(trace(Sigma))): {ate_proxy:.4f}")
    print(f"Minimum boundary margin: {safe_margin:.3f} cells")


if __name__ == "__main__":
    main()
