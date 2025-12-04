import sys
import os

# Sobe duas pastas (de evaluation -> alns -> algoritmo)
PASTA_ALGORITMO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PASTA_ALGORITMO)

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