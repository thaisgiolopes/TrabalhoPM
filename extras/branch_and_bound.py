from heuristics import precedence_ok_on_station, candidate_stations_limited

def branch_and_bound_refine(sol, removed_tasks):
    """
    B&B local nas tarefas removidas:
    - Explora reinserção viável (precedência/incapacidade) minimizando C parcial.
    - Usa poda por bound em C parcial.
    """
    inst = sol.inst
    ordered = sorted(removed_tasks, key=lambda i: (len(inst.pred[i]) + len(inst.succ[i])))
    best_C = float("inf")
    best_assignment = {}
    base_Ts = list(sol.Ts)

    def dfs(idx, current_Ts, partial_assignment):
        nonlocal best_C, best_assignment
        if idx == len(ordered):
            current_C = max(current_Ts)
            if current_C < best_C:
                best_C = current_C
                best_assignment = dict(partial_assignment)
            return
        i = ordered[idx]
        stations = candidate_stations_limited(sol, i)
        scored = []
        for s in stations:
            if not precedence_ok_on_station(sol, i, s):
                continue
            w = sol.worker_by_station[s-1]
            if i in inst.incapacidade[w]:
                continue
            t = inst.twi(w, i)
            new_load = current_Ts[s-1] + t
            scored.append((new_load, s, t))
        scored.sort(key=lambda x: x[0])
        for new_load, s, t in scored:
            trial_Ts = list(current_Ts)
            trial_Ts[s-1] = new_load
            trial_C = max(trial_Ts)
            if trial_C >= best_C:
                continue
            partial_assignment[i] = s
            dfs(idx + 1, trial_Ts, partial_assignment)
            del partial_assignment[i]

    dfs(0, base_Ts, {})

    if best_C == float("inf"):
        return False

    # aplica melhor reinserção
    for i in removed_tasks:
        sol.task_to_station[i] = None
    for i, s in best_assignment.items():
        sol.insert_task(i, s)
    sol.recompute_Ts_full()
    return True
