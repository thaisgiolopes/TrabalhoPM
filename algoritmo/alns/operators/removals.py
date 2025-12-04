import sys
import os

# Sobe duas pastas (de evaluation -> alns -> algoritmo)
PASTA_ALGORITMO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PASTA_ALGORITMO)

from solution import Solution
from evaluation.evaluate import evaluate_solution
import networkx as nx


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