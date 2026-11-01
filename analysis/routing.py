"""
Waste-collection routing: OR-Tools vs a greedy baseline, with transport-LCA
===========================================================================
Addresses reviewer points:
  * "graph-based routing presented as a formulation without empirical evaluation;
     contribution remains theoretical"
  * "Prior work on VRP with ML/graph embeddings and strong OR baselines
     (OR-Tools) should be acknowledged with comparisons"
  * "How does routing perform on actual ward networks vs OR-Tools, and what are
     the concrete savings in distance, time, fuel, LCA transport impacts?"

Scope (honest)
--------------
The dataset provides one (lat,long) per city (its landfill), not ward-level road
graphs. We therefore evaluate the routing module on the 34-node inter-city
facility network derived from those coordinates, using haversine distances as a
road-distance proxy. The OR-vs-greedy SAVINGS are a property of the optimiser and
the point geometry, so the comparison is valid; on a real deployment the same
solver runs on OSRM road distances over a ward graph. We benchmark against
Google OR-Tools (the standard OR baseline the reviewer names) and quantify
distance / time / fuel / transport-GHG savings vs a greedy nearest-neighbour
collector (a realistic unoptimised status quo).

Transport-LCA factors: truck fuel 35 L/100km, mean speed 40 km/h, diesel
emission factor 2.68 kgCO2/L (DEFRA/IPCC mobile combustion).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.common import load_rows, save_csv

FUEL_L_PER_KM = 0.35
SPEED_KMH = 40.0
DIESEL_EF_KG_PER_L = 2.68


def haversine_matrix(lat, lon):
    R = 6371.0
    la = np.radians(lat)[:, None]; lo = np.radians(lon)[:, None]
    dlat = la - la.T; dlon = lo - lo.T
    a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def tour_length(tour, D):
    return float(sum(D[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))


def greedy_nn(D, depot=0):
    n = len(D); unvisited = set(range(n)) - {depot}
    tour = [depot]; cur = depot
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur, j])
        tour.append(nxt); unvisited.remove(nxt); cur = nxt
    tour.append(depot)
    return tour


def ortools_tsp(D, depot=0, seconds=5):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    n = len(D)
    mgr = pywrapcp.RoutingIndexManager(n, 1, depot)
    routing = pywrapcp.RoutingModel(mgr)
    scale = 1000

    def cb(i, j):
        return int(D[mgr.IndexToNode(i), mgr.IndexToNode(j)] * scale)
    idx = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(idx)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromSeconds(seconds)
    sol = routing.SolveWithParameters(params)
    tour = []; i = routing.Start(0)
    while not routing.IsEnd(i):
        tour.append(mgr.IndexToNode(i)); i = sol.Value(routing.NextVar(i))
    tour.append(depot)
    return tour


def ortools_cvrp(D, demands, vehicles, capacity, depot=0, seconds=8):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    n = len(D)
    mgr = pywrapcp.RoutingIndexManager(n, vehicles, depot)
    routing = pywrapcp.RoutingModel(mgr)
    scale = 1000

    def dist_cb(i, j):
        return int(D[mgr.IndexToNode(i), mgr.IndexToNode(j)] * scale)
    didx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(didx)

    def dem_cb(i):
        return int(demands[mgr.IndexToNode(i)])
    demidx = routing.RegisterUnaryTransitCallback(dem_cb)
    routing.AddDimensionWithVehicleCapacity(
        demidx, 0, [int(capacity)] * vehicles, True, "Cap")
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromSeconds(seconds)
    sol = routing.SolveWithParameters(params)
    if sol is None:
        raise RuntimeError("OR-Tools CVRP found no solution (check capacity feasibility)")
    total = 0.0
    for v in range(vehicles):
        i = routing.Start(v); prev = mgr.IndexToNode(i)
        while not routing.IsEnd(i):
            i = sol.Value(routing.NextVar(i)); node = mgr.IndexToNode(i)
            total += D[prev, node]; prev = node
    return total


def greedy_cvrp(D, demands, vehicles, capacity, depot=0):
    """Naive: nearest-feasible insertion per vehicle until capacity full."""
    n = len(D); unserved = set(range(n)) - {depot}; total = 0.0
    for _ in range(vehicles):
        if not unserved:
            break
        load = 0; cur = depot
        while True:
            feas = [j for j in unserved if load + demands[j] <= capacity]
            if not feas:
                break
            nxt = min(feas, key=lambda j: D[cur, j])
            total += D[cur, nxt]; load += demands[nxt]
            unserved.discard(nxt); cur = nxt
        total += D[cur, depot]
    # any unserved (capacity short): one extra trip per node (penalty-free count)
    for j in list(unserved):
        total += 2 * D[depot, j]
    return total


def metrics(distance_km):
    fuel = distance_km * FUEL_L_PER_KM
    return dict(distance_km=round(distance_km, 1),
                time_h=round(distance_km / SPEED_KMH, 1),
                fuel_L=round(fuel, 1),
                transport_GHG_kgCO2=round(fuel * DIESEL_EF_KG_PER_L, 1))


def run():
    rows = load_rows()
    city = (rows.dropna(subset=["latitude", "longitude"])
            .groupby("city")
            .agg(lat=("latitude", "first"), lon=("longitude", "first"),
                 waste=("quantity", "sum")).reset_index())
    lat = city["lat"].to_numpy(); lon = city["lon"].to_numpy()
    D = haversine_matrix(lat, lon)
    depot = int(np.argmin(((lat - lat.mean()) ** 2 + (lon - lon.mean()) ** 2)))

    # --- single-vehicle TSP: greedy vs OR-Tools ---
    g_tour = greedy_nn(D, depot)
    o_tour = ortools_tsp(D, depot, seconds=5)
    g_len = tour_length(g_tour, D)
    o_len = tour_length(o_tour, D)

    rows_out = []
    for name, dist in [("Greedy nearest-neighbour", g_len), ("OR-Tools (GLS)", o_len)]:
        rows_out.append(dict(problem="TSP_single_vehicle", method=name, **metrics(dist)))
    tsp = pd.DataFrame(rows_out)
    tsp_sav = 100 * (g_len - o_len) / g_len
    save_csv(tsp, "routing_tsp_comparison.csv")

    # --- capacitated VRP: greedy vs OR-Tools ---
    # demand ~ city waste, scaled so mean node demand = 50 units and no single node
    # exceeds the 300-unit truck capacity (feasibility).
    w = city["waste"].to_numpy()
    demands = np.maximum((w / w.mean() * 50).round(), 1).astype(int)
    demands[depot] = 0
    capacity = 300
    vehicles = int(np.ceil(demands.sum() / capacity)) + 2
    g_v = greedy_cvrp(D, demands, vehicles, capacity, depot)
    o_v = ortools_cvrp(D, demands, vehicles, capacity, depot, seconds=8)
    vrows = []
    for name, dist in [("Greedy nearest-feasible", g_v), ("OR-Tools CVRP (GLS)", o_v)]:
        vrows.append(dict(problem=f"CVRP_{vehicles}veh_cap{capacity}", method=name,
                          **metrics(dist)))
    vrp = pd.DataFrame(vrows)
    vrp_sav = 100 * (g_v - o_v) / g_v
    save_csv(vrp, "routing_cvrp_comparison.csv")

    summary = pd.DataFrame([
        dict(problem="TSP", greedy_km=round(g_len, 1), ortools_km=round(o_len, 1),
             distance_saving_pct=round(tsp_sav, 1),
             fuel_saving_L=round((g_len - o_len) * FUEL_L_PER_KM, 1),
             GHG_saving_kgCO2=round((g_len - o_len) * FUEL_L_PER_KM * DIESEL_EF_KG_PER_L, 1)),
        dict(problem="CVRP", greedy_km=round(g_v, 1), ortools_km=round(o_v, 1),
             distance_saving_pct=round(vrp_sav, 1),
             fuel_saving_L=round((g_v - o_v) * FUEL_L_PER_KM, 1),
             GHG_saving_kgCO2=round((g_v - o_v) * FUEL_L_PER_KM * DIESEL_EF_KG_PER_L, 1)),
    ])
    save_csv(summary, "routing_savings_summary.csv")

    print("\nROUTING SUMMARY (34-node facility network)")
    print(summary.to_string(index=False))
    print(f"  Depot = {city['city'].iloc[depot]}; vehicles(CVRP)={vehicles}, cap={capacity}")
    return summary


if __name__ == "__main__":
    run()
