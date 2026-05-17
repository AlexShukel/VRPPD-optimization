#!/usr/bin/env python3
"""
CP-SAT driver for vrppd-or-tools: reads a JSON SolverRequest from stdin,
runs ortools.sat.python.cp_model, writes a JSON SolverResponse to stdout.
"""

import json
import math
import sys
import time

EARTH_RADIUS_KM = 6371.0
DIST_SCALE = 1_000_000


def great_circle_km(lat1, lon1, lat2, lon2):
    rlat1 = math.radians(lat1)
    rlon1 = math.radians(lon1)
    rlat2 = math.radians(lat2)
    rlon2 = math.radians(lon2)
    val = (
        math.sin(rlat1) * math.sin(rlat2)
        + math.cos(rlat1) * math.cos(rlat2) * math.cos(rlon1 - rlon2)
    )
    clamped = max(-1.0, min(1.0, val))
    return math.acos(clamped) * EARTH_RADIUS_KM


def build_geometry(req):
    V = len(req["problem"]["vehicles"])
    N = len(req["problem"]["orders"])
    coords = []
    for v in req["problem"]["vehicles"]:
        coords.append((v["start_lat"], v["start_lon"]))
    for o in req["problem"]["orders"]:
        coords.append((o["pickup_lat"], o["pickup_lon"]))
    for o in req["problem"]["orders"]:
        coords.append((o["delivery_lat"], o["delivery_lon"]))

    def dist_km(i, j):
        if i == j:
            return 0.0
        return great_circle_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])

    return {"V": V, "N": N, "num_nodes": len(coords), "dist_km": dist_km}


def _objective_weight(req, v):
    target = req["objective"]
    if target == "DISTANCE":
        return 1.0
    if target == "PRICE":
        return req["problem"]["vehicles"][v]["price_km"]
    raise ValueError(f"unsupported objective: {target}")


def self_test():
    import ortools
    from ortools.sat.python import cp_model  # noqa: F401

    print(f"ortools version: {ortools.__version__}")
    print("cp_model: OK")
    return 0


def fail(error_kind, error_msg):
    sys.stdout.write(json.dumps({
        "ok": False,
        "error_kind": error_kind,
        "error_msg": error_msg,
    }))
    sys.stdout.flush()
    sys.exit(1)


def succeed(objective_value, status, solver_runtime_ms):
    sys.stdout.write(json.dumps({
        "ok": True,
        "objective_value": float(objective_value),
        "status": status,
        "solver_runtime_ms": int(solver_runtime_ms),
    }))
    sys.stdout.flush()


def solve_cp_sat(req):
    from ortools.sat.python import cp_model

    started = time.monotonic()
    g = build_geometry(req)
    V, N = g["V"], g["N"]

    def start(v):
        return v

    def pickup(o):
        return V + o

    def delivery(o):
        return V + N + o

    def vehicle_nodes(v):
        return [start(v)] + [V + i for i in range(2 * N)]

    def is_pickup(node):
        if V <= node < V + N:
            return node - V
        return None

    def is_delivery(node):
        if V + N <= node < V + 2 * N:
            return node - V - N
        return None

    model = cp_model.CpModel()

    y = {}
    for o in range(N):
        for v in range(V):
            y[(o, v)] = model.NewBoolVar(f"y_{o}_{v}")

    x = {}
    obj_coeffs = []
    for v in range(V):
        nodes = vehicle_nodes(v)
        weight = _objective_weight(req, v)
        s = start(v)
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                xv = model.NewBoolVar(f"x_{i}_{j}_{v}")
                x[(i, j, v)] = xv
                cost = 0 if j == s else int(round(g["dist_km"](i, j) * weight * DIST_SCALE))
                if cost != 0:
                    obj_coeffs.append((xv, cost))

    q = {}
    for v in range(V):
        q[(start(v), v)] = model.NewIntVar(0, 0, f"q_start_{v}")
        for i in [V + k for k in range(2 * N)]:
            q[(i, v)] = model.NewIntVar(0, DIST_SCALE, f"q_{i}_{v}")

    u = {}
    M_u = 2 * N
    for v in range(V):
        for i in [V + k for k in range(2 * N)]:
            u[(i, v)] = model.NewIntVar(0, M_u, f"u_{i}_{v}")

    for o in range(N):
        model.Add(sum(y[(o, v)] for v in range(V)) == 1)

    for v in range(V):
        s = start(v)
        terms = [x[(s, j, v)] for j in [V + k for k in range(2 * N)] if (s, j, v) in x]
        if terms:
            model.Add(sum(terms) <= 1)

    for o in range(N):
        for v in range(V):
            p = pickup(o)
            d = delivery(o)
            nodes = vehicle_nodes(v)
            into_p = [x[(k, p, v)] for k in nodes if k != p and (k, p, v) in x]
            out_p = [x[(p, k, v)] for k in nodes if k != p and (p, k, v) in x]
            into_d = [x[(k, d, v)] for k in nodes if k != d and (k, d, v) in x]
            out_d = [x[(d, k, v)] for k in nodes if k != d and (d, k, v) in x]
            for row in (into_p, out_p, into_d, out_d):
                model.Add(sum(row) == y[(o, v)])

    for o in range(N):
        for v in range(V):
            p = pickup(o)
            d = delivery(o)
            model.Add(u[(p, v)] - u[(d, v)] + M_u * y[(o, v)] <= M_u - 1)

    M_q = 2 * DIST_SCALE
    for v in range(V):
        nodes = vehicle_nodes(v)
        s = start(v)
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                if is_pickup(j) is None and is_delivery(j) is None:
                    continue
                xij = x[(i, j, v)]
                expr = q[(j, v)]
                if i != s:
                    expr = expr - q[(i, v)]
                ip = is_pickup(i)
                id_ = is_delivery(i)
                if ip is not None:
                    w = int(round(DIST_SCALE / req["problem"]["orders"][ip]["load_factor"]))
                    expr = expr - w * y[(ip, v)]
                elif id_ is not None:
                    w = int(round(DIST_SCALE / req["problem"]["orders"][id_]["load_factor"]))
                    expr = expr + w * y[(id_, v)]
                model.Add(expr - M_q * xij >= -M_q)
                model.Add(expr + M_q * xij <= M_q)

    M_n = 2 * N
    svc = [V + k for k in range(2 * N)]
    for v in range(V):
        for i in svc:
            for j in svc:
                if i == j:
                    continue
                if (i, j, v) in x:
                    model.Add(u[(i, v)] - u[(j, v)] + M_n * x[(i, j, v)] <= M_n - 1)

    model.Minimize(sum(coef * var for var, coef in obj_coeffs))

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = max(1, req["threads"])
    solver.parameters.max_time_in_seconds = max(0.001, req["timeout_ms"] / 1000.0)
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)

    status_str = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "FAILED",
        cp_model.UNKNOWN: "TIMED_OUT",
    }.get(status, "FAILED")

    value = solver.ObjectiveValue() / DIST_SCALE if status_str in ("OPTIMAL", "FEASIBLE") else 0.0
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return succeed(value, status_str, elapsed_ms)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        return self_test()

    try:
        from ortools.sat.python import cp_model  # noqa: F401
    except ImportError as e:
        fail("ortools_import", str(e))

    try:
        req = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        fail("invalid_request", f"stdin JSON parse: {e}")

    solve_cp_sat(req)


if __name__ == "__main__":
    sys.exit(main() or 0)
