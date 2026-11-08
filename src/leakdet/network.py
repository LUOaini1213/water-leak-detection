"""Minimal EPANET .inp reader and graph utilities (no hydraulic simulation).

Only the sections needed for topology, tank geometry, nominal demands and
distances are parsed. Units follow the file (L-Town uses CMH: m3/h, metres).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


@dataclass
class Network:
    junctions: dict[str, float] = field(default_factory=dict)  # id -> elevation
    reservoirs: list[str] = field(default_factory=list)
    tanks: dict[str, dict] = field(default_factory=dict)  # id -> geometry
    pipes: dict[str, tuple[str, str, float]] = field(default_factory=dict)  # id -> (n1, n2, length m)
    pumps: dict[str, tuple[str, str]] = field(default_factory=dict)
    valves: dict[str, tuple[str, str, str, float]] = field(default_factory=dict)  # (n1, n2, type, setting)
    # junction -> {pattern id: base demand}; from [DEMANDS] if present, else [JUNCTIONS]
    demands: dict[str, dict[str, float]] = field(default_factory=dict)
    patterns: dict[str, list[float]] = field(default_factory=dict)
    options: dict[str, str] = field(default_factory=dict)

    @property
    def nodes(self) -> list[str]:
        return list(self.junctions) + self.reservoirs + list(self.tanks)

    @property
    def base_demand(self) -> dict[str, float]:
        """Summed nominal base demand per junction (all categories)."""
        return {n: sum(d.values()) for n, d in self.demands.items()}

    def link_nodes(self, link: str) -> tuple[str, str]:
        if link in self.pipes:
            return self.pipes[link][:2]
        if link in self.pumps:
            return self.pumps[link]
        return self.valves[link][:2]

    def link_length(self, link: str) -> float:
        """Pipe length in metres; pumps and valves have length 0 (as in EPANET)."""
        return self.pipes[link][2] if link in self.pipes else 0.0


def _rows(text: str):
    section = None
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            section = line.strip("[]").upper()
            continue
        yield section, line.split()


def parse_inp(text: str) -> Network:
    net = Network()
    junction_demands: dict[str, dict[str, float]] = {}
    section_demands: dict[str, dict[str, float]] = {}
    for section, tok in _rows(text):
        if section == "JUNCTIONS":
            net.junctions[tok[0]] = float(tok[1])
            if len(tok) > 2:
                pat = tok[3] if len(tok) > 3 else ""
                junction_demands.setdefault(tok[0], {})[pat] = float(tok[2])
        elif section == "RESERVOIRS":
            net.reservoirs.append(tok[0])
        elif section == "TANKS":
            net.tanks[tok[0]] = {
                "elevation": float(tok[1]),
                "init_level": float(tok[2]),
                "min_level": float(tok[3]),
                "max_level": float(tok[4]),
                "diameter": float(tok[5]),
                "min_vol": float(tok[6]) if len(tok) > 6 else 0.0,
                "vol_curve": tok[7] if len(tok) > 7 and tok[7] != "*" else None,
            }
        elif section == "PIPES":
            net.pipes[tok[0]] = (tok[1], tok[2], float(tok[3]))
        elif section == "PUMPS":
            net.pumps[tok[0]] = (tok[1], tok[2])
        elif section == "VALVES":
            net.valves[tok[0]] = (tok[1], tok[2], tok[4].upper(), float(tok[5]))
        elif section == "DEMANDS":
            pat = tok[2] if len(tok) > 2 else ""
            node = section_demands.setdefault(tok[0], {})
            node[pat] = node.get(pat, 0.0) + float(tok[1])
        elif section == "PATTERNS":
            net.patterns.setdefault(tok[0], []).extend(float(v) for v in tok[1:])
        elif section == "OPTIONS":
            net.options[" ".join(tok[:-1]).upper()] = tok[-1]
    # [DEMANDS] overrides the single demand column of [JUNCTIONS] (EPANET rule)
    net.demands = section_demands if section_demands else junction_demands
    return net


def load_network(path: str | Path) -> Network:
    path = Path(path)
    if path.suffix == ".gz":
        import gzip

        return parse_inp(gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace"))
    return parse_inp(path.read_text(encoding="utf-8", errors="replace"))


def tank_area_m2(tank: dict) -> float:
    """Plan area of a cylindrical tank (no volume curve support)."""
    if tank.get("vol_curve"):
        raise ValueError("volume curves are not supported")
    return math.pi * tank["diameter"] ** 2 / 4.0


def _index(net: Network) -> dict[str, int]:
    return {n: i for i, n in enumerate(net.nodes)}


def pipe_graph(net: Network, exclude: set[str] | None = None):
    """Sparse symmetric adjacency weighted by pipe length.

    Mirrors the BattLeDIM scoring code: links with zero length (pumps, valves)
    do not create edges. Links in ``exclude`` are skipped as well.
    """
    idx = _index(net)
    best: dict[tuple[int, int], float] = {}
    for pid, (a, b, length) in net.pipes.items():
        if (exclude and pid in exclude) or length <= 0:
            continue
        key = (min(idx[a], idx[b]), max(idx[a], idx[b]))
        best[key] = min(length, best.get(key, math.inf))
    rows = [k[0] for k in best] + [k[1] for k in best]
    cols = [k[1] for k in best] + [k[0] for k in best]
    vals = list(best.values()) * 2
    n = len(idx)
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr(), idx


def node_distances(net: Network, sources: list[str] | None = None) -> tuple[np.ndarray, dict[str, int]]:
    """Shortest pipe-length distance (m) from ``sources`` (rows) to all nodes (columns)."""
    g, idx = pipe_graph(net)
    src = None if sources is None else [idx[s] for s in sources]
    return dijkstra(g, directed=False, indices=src), idx


def link_distance(net: Network, dist: np.ndarray, idx: dict[str, int], l1: str, l2: str) -> float:
    """BattLeDIM link-to-link distance: closest pair of end nodes plus half of each length.

    ``dist`` must be the all-pairs node distance matrix from :func:`node_distances`.
    """
    if l1 == l2:
        return 0.0
    a1, b1 = net.link_nodes(l1)
    a2, b2 = net.link_nodes(l2)
    best = min(dist[idx[u], idx[v]] for u in (a1, b1) for v in (a2, b2))
    return float(best + net.link_length(l1) / 2 + net.link_length(l2) / 2)


def hydraulic_areas(net: Network) -> dict[str, str]:
    """Label every node with the supply area it belongs to.

    Pumps and valves are boundaries (they create no pipe edges). The component
    holding a tank is called after the tank's area ("C" for L-Town's T1 zone);
    the largest component is "A"; remaining components with demand are "B".. in
    order of size; components holding a reservoir are "source".
    """
    g, idx = pipe_graph(net)
    _, labels = connected_components(g, directed=False)
    comp_nodes: dict[int, list[str]] = {}
    for n, i in idx.items():
        comp_nodes.setdefault(int(labels[i]), []).append(n)
    names: dict[int, str] = {}
    tank_comps = {int(labels[idx[t]]) for t in net.tanks}
    res_comps = {int(labels[idx[r]]) for r in net.reservoirs}
    for c in res_comps:
        names[c] = "source"
    for c in tank_comps:
        names[c] = "C"
    rest = sorted((c for c in comp_nodes if c not in names), key=lambda c: -len(comp_nodes[c]))
    for letter, c in zip("ABDEFGH", rest):
        names[c] = letter
    return {n: names[int(labels[i])] for n, i in idx.items()}


def pattern_value(net: Network, pattern: str, step: np.ndarray) -> np.ndarray:
    """Multiplier of ``pattern`` at integer pattern steps (cyclic); '' means 1.0."""
    values = net.patterns.get(pattern)
    if not values:
        return np.ones(len(step))
    return np.asarray(values)[np.asarray(step) % len(values)]


def nominal_demand(net: Network, nodes: list[str], step: np.ndarray) -> np.ndarray:
    """Nominal (model) demand summed over ``nodes`` at pattern steps, in model flow units."""
    total = np.zeros(len(step))
    per_pattern: dict[str, float] = {}
    for n in nodes:
        for pat, base in net.demands.get(n, {}).items():
            per_pattern[pat] = per_pattern.get(pat, 0.0) + base
    mult = float(net.options.get("DEMAND MULTIPLIER", 1.0))
    for pat, base in per_pattern.items():
        total += base * pattern_value(net, pat, step)
    return total * mult
