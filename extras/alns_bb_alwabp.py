import sys
from instance import Instance
from alns_core import ALNS
from branch_and_bound import branch_and_bound_refine
import time

def run_alns_bb(instancia_path, seeds=(42, 123, 777, 2025, 9999),
                max_iter=3000, time_limit=120, segment_length=100,
                cooling=0.995, regret_k=2, remove_low=0.2, remove_high=0.4,
                patience=800):
    inst = Instance(instancia_path)
    results = []
    sols = []

    for seed in seeds:
        start_seed = time.time()
        alns = ALNS(inst,
                    seed=seed,
                    segment_length=segment_length,
                    cooling=cooling,
                    max_iter=max_iter,
                    regret_k=regret_k,
                    remove_frac=(remove_low, remove_high),
                    time_limit=time_limit,
                    patience=patience,
                    refine_callback=branch_and_bound_refine)
        try:
            best = alns.run(seed_init=seed)
        except RuntimeError as e:
            print(f"Seed {seed}: falhou ({e})")
            continue

        elapsed = time.time() - start_seed
        if time_limit and elapsed > time_limit:
            print(f"Seed {seed}: excedeu limite de tempo ({elapsed:.1f}s), pulando para próximo seed.")
            continue

        results.append(best.C)
        sols.append(best)
        print(f"Seed {seed}: melhor C = {best.C:.6f} (tempo {elapsed:.1f}s)")

    mean_C = sum(results) / len(results)
    best_C = min(results)
    worst_C = max(results)
    print("\nResumo das 5 execuções:")
    print(f"Média C: {mean_C:.6f}")
    print(f"Melhor C: {best_C:.6f}")
    print(f"Pior C: {worst_C:.6f}")

    idx_best = results.index(best_C)
    best_sol = sols[idx_best]
    print("\nSolução da melhor execução:")
    for s_idx in range(inst.m):
        w = best_sol.worker_by_station[s_idx]
        tarefas = sorted(best_sol.tasks_by_station[s_idx])
        Ts = best_sol.Ts[s_idx]
        print(f"Estação {s_idx+1}: trabalhador {w}, T_s = {Ts:.6f}, tarefas: {tarefas}")

    return best_sol

if __name__ == "__main__":
    run_alns_bb("instancias/1_ton")
