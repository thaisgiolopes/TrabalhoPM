# -------------------------
# Avaliação e viabilidade
# -------------------------

import math
import random
import networkx as nx
from alwabpData import ALWABPData
from solution import Solution

def compute_station_load(data: ALWABPData, station_tasks, worker):
    # Soma dos tempos das tarefas na estação considerando o trabalhador
    total = 0.0
    for i in station_tasks:
        t = data.twi(worker, i)
        if t == float("inf"):
            return float("inf")  # inviável
        total += t
    return total

def evaluate_solution(sol: Solution):
    data = sol.data
    # Recalcular tempos por estação
    sol.Ts = []
    for s_idx in range(data.m):
        worker = sol.worker_by_station[s_idx]
        if worker is None:
            sol.Ts.append(float("inf"))
            continue
        load = compute_station_load(data, sol.tasks_by_station[s_idx], worker)
        sol.Ts.append(load)
    sol.C = max(sol.Ts) if sol.Ts else float("inf")
    return sol.C

def respects_precedence(data: ALWABPData, tasks_by_station):
    # Verifica se para todo (i,j) temos station(i) <= station(j)
    pos_station = {}
    for s_idx, tasks in enumerate(tasks_by_station, start=1):  # estações 1..m
        for i in tasks:
            pos_station[i] = s_idx
    for (i, j) in data.precedencias:
        si = pos_station.get(i)
        sj = pos_station.get(j)
        if si is None or sj is None:
            return False
        if si > sj:
            return False
    return True

def feasible_insertion_station(data: ALWABPData, tasks_by_station, task, station_idx):
    # Checa precedência se task fosse alocada na estação station_idx
    # Basta garantir que para todo predecessor p de task, station(p) <= station_idx
    # e para todo sucessor s de task, station_idx <= station(s)
    pred = list(data.G.predecessors(task))
    succ = list(data.G.successors(task))
    pos_station = {} # Map task -> station_idx
    # Reconstruct pos_station from the current tasks_by_station state
    for s, lst in enumerate(tasks_by_station, start=1):
        for i in lst:
            pos_station[i] = s

    for p in pred:
        sp = pos_station.get(p)
        if sp is None: # Predecessor not yet placed in a station
            # This case means the predecessor must be part of tasks_to_insert
            # or is yet to be considered. We assume it will be placed correctly.
            # For now, it doesn't violate precedence *unless* the predecessor
            # is placed in a station *after* the current station_idx.
            # This function only checks for existing tasks.
            continue
        if sp > station_idx:
            return False # Predecessor is in a later station

    for s in succ:
        ss = pos_station.get(s)
        if ss is None:
            # Similar to predecessors, if a successor is not yet placed,
            # we assume it will be placed correctly later.
            continue
        if station_idx > ss:
            return False # Successor is in an earlier station
    return True

# -------------------------
# Construção de solução inicial
# -------------------------

def initial_solution_greedy(data: ALWABPData, seed=0):
    random.seed(seed)
    sol = Solution(data)

    # Atribui trabalhadores às estações inicialmente (ex.: identidade, ou shuffle para diversificação)
    workers = list(data.trabalhadores)
    random.shuffle(workers)
    sol.worker_by_station = workers

    # Ordenação topológica de tarefas
    order = list(nx.topological_sort(data.G))

    # Heurística: para cada tarefa na ordem, colocar na menor estação viável que minimize o aumento em C
    for i in order:
        best_s = None
        best_cost = float("inf")
        for s_idx in range(1, data.m+1):
            if not feasible_insertion_station(data, sol.tasks_by_station, i, s_idx):
                continue
            w = sol.worker_by_station[s_idx-1]
            if w is None: # Defensive check, though workers should be assigned
                continue
            if i in data.incapacidade[w]:
                continue
            # custo estimado: carga da estação após inserir
            current_load = compute_station_load(data, sol.tasks_by_station[s_idx-1], w)
            if current_load == float("inf"): # If station is already inviable
                continue
            twi_val = data.twi(w, i)
            if twi_val == float("inf"): # Should be caught by incapacity, but double check
                continue
            new_load = current_load + twi_val
            # proxy do novo C após inserir
            proxy_C = max(new_load, max(sol.Ts) if sol.Ts else 0.0)
            if proxy_C < best_cost:
                best_cost = proxy_C
                best_s = s_idx
        if best_s is None:
            # If no station found, try fallback (this part remains as original, assuming it finds one)
            # fallback: colocar na última estação viável (se existir)
            for s_idx in range(data.m, 0, -1):
                if feasible_insertion_station(data, sol.tasks_by_station, i, s_idx):
                    w = sol.worker_by_station[s_idx-1]
                    if w is None or i in data.incapacidade[w]:
                        continue
                    best_s = s_idx
                    break
        if best_s is None:
            # não foi possível inserir — instância ou arranjo de workers inviável; tentar reembaralhar workers
            return None

        sol.tasks_by_station[best_s-1].append(i)
        evaluate_solution(sol)

    evaluate_solution(sol)
    if not respects_precedence(data, sol.tasks_by_station):
        return None
    return sol

# -------------------------
# Operadores de remoção
# -------------------------

def remove_random(sol: Solution, q, rng):
    removed = []
    all_tasks = [i for lst in sol.tasks_by_station for i in lst]
    rng.shuffle(all_tasks)
    for i in all_tasks[:q]:
        # remove i da estação corrente
        for s_idx in range(len(sol.tasks_by_station)): # Iterate through original indices
            if i in sol.tasks_by_station[s_idx]:
                sol.tasks_by_station[s_idx].remove(i)
                removed.append(i)
                break
    evaluate_solution(sol)
    return removed

def remove_worst(sol: Solution, q, rng):
    data = sol.data
    # "Custo" da tarefa = tempo da tarefa no trabalhador atual + impacto na estação gargalo
    contrib = []
    for s_idx in range(len(sol.tasks_by_station)):
        w = sol.worker_by_station[s_idx]
        if w is None: # Skip if worker not assigned to station (should not happen after init)
            continue
        for i in sol.tasks_by_station[s_idx]:
            # If twi is inf, it means incapacity, which should have been handled. Defensive check:
            twi_val = data.twi(w, i)
            if twi_val == float("inf"):
                continue # Skip task if worker is incapable (should not be in solution anyway)
            contrib.append((twi_val, i)) # Only consider tasks with finite time
    contrib.sort(reverse=True)  # maiores primeiros
    removed = []
    for _, i in contrib[:q]:
        for s_idx in range(len(sol.tasks_by_station)):
            if i in sol.tasks_by_station[s_idx]:
                sol.tasks_by_station[s_idx].remove(i)
                removed.append(i)
                break
    evaluate_solution(sol)
    return removed

def remove_shaw(sol: Solution, q, rng):
    # Similaridade baseada em precedência proximidade e tempo similar na estação atual
    data = sol.data
    all_tasks = [i for lst in sol.tasks_by_station for i in lst]
    if not all_tasks:
        return []
    seed_task = rng.choice(all_tasks)
    def relatedness(a, b):
        # proximidade no grafo (distância de precedência) e diferença de tempo médio
        try:
            dist = nx.shortest_path_length(data.G.to_undirected(), source=a, target=b)
        except nx.NetworkXNoPath:
            dist = 10 # Default large distance if no path

        # Get average time for task 'a' and 'b' across all capable workers
        ta_sum = sum(data.tempos[a-1][w-1] for w in data.trabalhadores if data.tempos[a-1][w-1] != float("inf"))
        na_count = sum(1 for w in data.trabalhadores if data.tempos[a-1][w-1] != float("inf"))
        ma = ta_sum / max(na_count, 1) if na_count > 0 else 0.0 # Handle division by zero

        tb_sum = sum(data.tempos[b-1][w-1] for w in data.trabalhadores if data.tempos[b-1][w-1] != float("inf"))
        nb_count = sum(1 for w in data.trabalhadores if data.tempos[b-1][w-1] != float("inf"))
        mb = tb_sum / max(nb_count, 1) if nb_count > 0 else 0.0 # Handle division by zero

        return dist + abs(ma - mb) / max(ma + mb, 1.0)

    # Rank tasks by relatedness to the seed task
    # Ensure tasks are only from all_tasks to avoid KeyError for non-existent tasks
    ranked = sorted(all_tasks, key=lambda t: relatedness(seed_task, t))
    removed = []
    for i in ranked[:q]:
        for s_idx in range(len(sol.tasks_by_station)): # Iterate through original indices
            if i in sol.tasks_by_station[s_idx]:
                sol.tasks_by_station[s_idx].remove(i)
                removed.append(i)
                break
    evaluate_solution(sol)
    return removed

# -------------------------
# Operadores de inserção
# -------------------------

def insert_greedy(sol: Solution, tasks_to_insert, rng, noise=0.0):
    data = sol.data
    if not tasks_to_insert: # If nothing to insert, it's a success
        return True

    for i in tasks_to_insert:
        best_s = None
        best_cost = float("inf")
        for s_idx in range(1, data.m+1):
            if not feasible_insertion_station(data, sol.tasks_by_station, i, s_idx):
                continue
            w = sol.worker_by_station[s_idx-1]
            if w is None: # Defensive check
                continue
            if i in data.incapacidade[w]:
                continue
            current_load = compute_station_load(data, sol.tasks_by_station[s_idx-1], w)
            if current_load == float("inf"): # If station is already inviable
                continue
            twi_val = data.twi(w, i)
            if twi_val == float("inf"): # Should be caught by incapacity, but double check
                continue
            new_load = current_load + twi_val
            proxy_C = max(new_load, max(sol.Ts) if sol.Ts else 0.0)
            proxy_C += noise * rng.random()
            if proxy_C < best_cost:
                best_cost = proxy_C
                best_s = s_idx
        if best_s is None: # If a task cannot be inserted anywhere, this insertion failed
            return False
        sol.tasks_by_station[best_s-1].append(i)
        evaluate_solution(sol)
    return True # All tasks were successfully inserted

def insert_regret(sol: Solution, tasks_to_insert, rng, k=2, noise=0.0):
    data = sol.data
    pending = set(tasks_to_insert)
    if not pending: # If nothing to insert, it's a success
        return True

    while pending:
        # para cada tarefa, calcule as k melhores posições
        best_for_task = {}
        for i in list(pending):
            options = []
            for s_idx in range(1, data.m+1):
                if not feasible_insertion_station(data, sol.tasks_by_station, i, s_idx):
                    continue
                w = sol.worker_by_station[s_idx-1]
                if w is None: # Defensive check
                    continue
                if i in data.incapacidade[w]:
                    continue
                current_load = compute_station_load(data, sol.tasks_by_station[s_idx-1], w)
                if current_load == float("inf"): # If station is already inviable
                    continue
                twi_val = data.twi(w, i)
                if twi_val == float("inf"): # Should be caught by incapacity, but double check
                    continue
                new_load = current_load + twi_val
                proxy_C = max(new_load, max(sol.Ts) if sol.Ts else 0.0)
                proxy_C += noise * rng.random()
                options.append((proxy_C, s_idx))
            options.sort()
            if options:
                best_for_task[i] = options[:max(k,1)]

        if not best_for_task: # If no feasible insertion for any pending task, the whole reconstruction fails
            return False

        chosen_task = None
        chosen_station = None
        best_regret = -float("inf")
        for i, opts in best_for_task.items():
            if len(opts) == 1:
                regret = float("inf") # Only one option, so regret is infinite (very attractive)
            else:
                regret = opts[k-1][0] - opts[0][0] # Difference between k-th best and best option
            if regret > best_regret: # Select task with highest regret
                best_regret = regret
                chosen_task = i
                chosen_station = opts[0][1] # The station for the best option

        # Defensive check: if for some reason chosen_task or chosen_station are still None here,
        # something went wrong in the selection logic above, so we return False.
        if chosen_task is None or chosen_station is None:
            return False

        # insere escolhida
        sol.tasks_by_station[chosen_station-1].append(chosen_task)
        evaluate_solution(sol)
        pending.remove(chosen_task)
    return True # All tasks were successfully inserted

# -------------------------
# Operadores sobre trabalhadores
# -------------------------

def swap_workers(sol: Solution, rng):
    data = sol.data
    if data.m < 2: # Cannot swap if fewer than 2 stations
        return False
    a, b = rng.sample(range(data.m), 2)
    sol.worker_by_station[a], sol.worker_by_station[b] = sol.worker_by_station[b], sol.worker_by_station[a]
    evaluate_solution(sol)
    return True

def reassign_worst_station_worker(sol: Solution, rng):
    data = sol.data
    if not sol.Ts or data.m < 2: # Cannot reassign if no station times or fewer than 2 stations
        return False
    # Find the index of the station with the highest cycle time (worst station)
    worst_idx = max(range(len(sol.Ts)), key=lambda s: sol.Ts[s])
    
    # Select a different station randomly to swap workers with
    other_stations = [i for i in range(len(sol.Ts)) if i != worst_idx]
    if not other_stations: # Should not happen if data.m >= 2
        return False
    other = rng.choice(other_stations)

    # Swap the workers between the worst station and the chosen other station
    sol.worker_by_station[worst_idx], sol.worker_by_station[other] = \
        sol.worker_by_station[other], sol.worker_by_station[worst_idx]
    evaluate_solution(sol)
    return True

# -------------------------
# Critério de aceitação (SA)
# -------------------------

def accept(old_cost, new_cost, T, rng):
    if new_cost < old_cost:
        return True
    delta = new_cost - old_cost
    prob = math.exp(-delta / max(T, 1e-9))
    return rng.random() < prob