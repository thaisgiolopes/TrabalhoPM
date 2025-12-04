import sys
import os

# Sobe duas pastas (de evaluation -> alns -> algoritmo)
PASTA_ALGORITMO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PASTA_ALGORITMO)

from solution import Solution
from evaluation.evaluate import compute_station_load, evaluate_solution
from evaluation.feasibility import feasible_insertion_station


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