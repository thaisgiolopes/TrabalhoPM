import sys
import os
# permite importar da pasta pai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import networkx as nx
from evaluation.evaluate import compute_station_load, evaluate_solution
from evaluation.feasibility import feasible_insertion_station, respects_precedence
from solution import Solution
from alwabpData import ALWABPData

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