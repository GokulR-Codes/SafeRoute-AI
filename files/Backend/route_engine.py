"""
Namma Safe BLR — Safe Route Engine
Modified Dijkstra / A* where edge cost = risk score (not distance).
The algorithm builds a synthetic road graph over Bangalore using a grid
mesh + major waypoints, then finds the path minimising cumulative risk.
"""

import heapq, math
import numpy as np
from typing import List, Tuple, Dict, Optional
from safety_score import segment_safety_score

# ─── Haversine distance (km) ──────────────────────────────────────────────────
def haversine(p1: tuple, p2: tuple) -> float:
    R = 6371.0
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


# ─── Graph construction ────────────────────────────────────────────────────────
def build_graph(
    src: tuple,
    dst: tuple,
    hour: int,
    grid_step: float = 0.008,      # ~0.9 km grid resolution
    max_dist_km: float = 1.2,      # max edge length
    safety_weight: float = 0.75,   # blend: 75% safety, 25% distance
) -> Dict[tuple, List[tuple]]:
    """
    Generates a grid-based navigation graph between src and dst.
    Nodes are lat/lon grid points; edges carry a blended cost.
    Returns adjacency list: node → [(cost, neighbour), ...]
    """
    min_lat = min(src[0], dst[0]) - 0.02
    max_lat = max(src[0], dst[0]) + 0.02
    min_lon = min(src[1], dst[1]) - 0.02
    max_lon = max(src[1], dst[1]) + 0.02

    # Grid nodes
    lats = np.arange(min_lat, max_lat, grid_step)
    lons = np.arange(min_lon, max_lon, grid_step)
    nodes = [(round(lat, 6), round(lon, 6)) for lat in lats for lon in lons]

    # Ensure source/destination are in the graph
    nodes += [src, dst]
    nodes = list(set(nodes))

    # Build adjacency list
    graph: Dict[tuple, List[tuple]] = {n: [] for n in nodes}

    for i, n1 in enumerate(nodes):
        for n2 in nodes[i+1:]:
            dist_km = haversine(n1, n2)
            if dist_km > max_dist_km:
                continue

            risk   = segment_safety_score(n1, n2, hour)
            dist_n = dist_km / 20.0          # normalise to [0,1] range

            # Blended edge cost
            cost = safety_weight * risk + (1 - safety_weight) * dist_n

            graph[n1].append((cost, n2))
            graph[n2].append((cost, n1))

    return graph


# ─── Dijkstra shortest path ────────────────────────────────────────────────────
def dijkstra(
    graph: Dict[tuple, List[tuple]],
    src: tuple,
    dst: tuple,
) -> Tuple[List[tuple], float]:
    """
    Standard Dijkstra on the risk graph.
    Returns (path_as_list_of_nodes, total_cost).
    """
    dist_map = {node: float("inf") for node in graph}
    dist_map[src] = 0.0
    prev_map: Dict[tuple, Optional[tuple]] = {node: None for node in graph}

    pq = [(0.0, src)]          # (cost, node)

    while pq:
        cost, u = heapq.heappop(pq)
        if cost > dist_map[u]:
            continue
        if u == dst:
            break
        for edge_cost, v in graph.get(u, []):
            new_cost = dist_map[u] + edge_cost
            if new_cost < dist_map[v]:
                dist_map[v] = new_cost
                prev_map[v] = u
                heapq.heappush(pq, (new_cost, v))

    # Reconstruct path
    path = []
    node = dst
    while node is not None:
        path.append(node)
        node = prev_map.get(node)
    path.reverse()

    if path[0] != src:
        return [], float("inf")           # no path found

    return path, dist_map[dst]


# ─── A* heuristic (geographic distance) ──────────────────────────────────────
def astar(
    graph: Dict[tuple, List[tuple]],
    src: tuple,
    dst: tuple,
    safety_weight: float = 0.75,
) -> Tuple[List[tuple], float]:
    """A* with geographic heuristic for faster convergence."""
    h = lambda n: haversine(n, dst) / 20.0 * (1 - safety_weight)

    open_set = [(h(src), 0.0, src)]
    g_score  = {src: 0.0}
    came_from: Dict[tuple, Optional[tuple]] = {src: None}

    while open_set:
        _, g, u = heapq.heappop(open_set)
        if u == dst:
            break
        if g > g_score.get(u, float("inf")):
            continue
        for edge_cost, v in graph.get(u, []):
            tentative_g = g_score[u] + edge_cost
            if tentative_g < g_score.get(v, float("inf")):
                g_score[v]   = tentative_g
                came_from[v] = u
                f = tentative_g + h(v)
                heapq.heappush(open_set, (f, tentative_g, v))

    path = []
    node: Optional[tuple] = dst
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    path.reverse()

    if not path or path[0] != src:
        return [], float("inf")

    return path, g_score.get(dst, float("inf"))


# ─── Public API ───────────────────────────────────────────────────────────────
def find_safe_route(
    src_lat: float, src_lon: float,
    dst_lat: float, dst_lon: float,
    hour:    int = 22,
    algorithm: str = "astar",
) -> dict:
    """
    Main entry point.  Returns the safest route as a GeoJSON-style dict.

    Returns
    -------
    {
        "coordinates": [[lat, lon], ...],
        "total_risk_score": float,
        "distance_km": float,
        "waypoint_scores": [float, ...],
        "risk_level": str,
    }
    """
    src = (round(src_lat, 6), round(src_lon, 6))
    dst = (round(dst_lat, 6), round(dst_lon, 6))

    print(f"🗺️  Building route graph  {src} → {dst}  hour={hour}")
    graph = build_graph(src, dst, hour)

    if algorithm == "astar":
        path, cost = astar(graph, src, dst)
    else:
        path, cost = dijkstra(graph, src, dst)

    if not path:
        return {"error": "No route found"}

    # Per-waypoint safety scores
    waypoint_scores = [
        round(segment_safety_score(path[i], path[i+1], hour), 4)
        for i in range(len(path) - 1)
    ]

    total_dist = sum(haversine(path[i], path[i+1]) for i in range(len(path)-1))
    avg_risk   = float(np.mean(waypoint_scores)) if waypoint_scores else cost

    if avg_risk < 0.30:   risk_lvl = "LOW"
    elif avg_risk < 0.55: risk_lvl = "MEDIUM"
    elif avg_risk < 0.75: risk_lvl = "HIGH"
    else:                 risk_lvl = "CRITICAL"

    print(f"✅ Route found: {len(path)} waypoints, dist={total_dist:.2f} km, risk={avg_risk:.3f}")

    return {
        "coordinates":    [list(p) for p in path],
        "total_risk_score": round(avg_risk, 4),
        "total_risk_pct":   round(avg_risk * 100, 1),
        "distance_km":      round(total_dist, 3),
        "waypoint_scores":  waypoint_scores,
        "risk_level":       risk_lvl,
        "algorithm":        algorithm,
        "waypoints":        len(path),
    }
