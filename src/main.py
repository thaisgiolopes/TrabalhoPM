import sys
import math
import random
import time
from collections import defaultdict, deque
from alns_class import ALNS
from alwabpData import ALWABPData
from ler_instancia import ler_instancia


# -------------------------
# CLI simples para executar ALNS
# -------------------------

def run_alns(instancia_path, seed=0, time_limit=300, max_iter=5000, segment_length=100,
             cooling=0.995, noise=0.0, regret_k=2, remove_frac_low=0.2, remove_frac_high=0.4):
    n, tempos, precedencias = ler_instancia(instancia_path)
    data = ALWABPData(n, tempos, precedencias)
    alns = ALNS(data,
                seed=seed,
                segment_length=segment_length,
                cooling=cooling,
                max_iter=max_iter,
                noise=noise,
                regret_k=regret_k,
                remove_frac=(remove_frac_low, remove_frac_high),
                time_limit=time_limit)
    best = alns.run(seed_init=seed)

    # Impressão de resultados
    if best is None: # Handle case where ALNS might return None if no valid solution found
        print("ALNS did not find a valid solution.")
        return None

    print(f"Tempo de ciclo (ALNS) C: {best.C:.6f}")
    for s_idx in range(data.m):
        w = best.worker_by_station[s_idx]
        tarefas = sorted(best.tasks_by_station[s_idx])
        Ts = best.Ts[s_idx]
        print(f"Estação {s_idx+1}: trabalhador {w}, T_s = {Ts:.6f}, tarefas: {tarefas}")

    return best

run_alns('instancias/1_hes')