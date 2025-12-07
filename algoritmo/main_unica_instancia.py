import os
import random
import sys
from alns.alns import ALNS
from algoritmo.alwabpData import ALWABPData
from algoritmo.ler_instancia import ler_instancia

def run_alns(instancia_path, time_limit=180, max_iter=5000, segment_length=100,
             cooling=0.995, noise=0.0, regret_k=2, remove_frac_low=0.2, remove_frac_high=0.4,
             output_file="resultado.txt"):
    n, tempos, precedencias = ler_instancia(instancia_path)
    data = ALWABPData(n, tempos, precedencias)

    # gera 5 seeds aleatórios fixos
    seeds = [random.randint(1, 1000) for _ in range(5)]

    with open(output_file, "w") as f:
        for seed in seeds:
            alns = ALNS(data,
                        seed=seed,
                        segment_length=segment_length,
                        cooling=cooling,
                        max_iter=max_iter,
                        noise=noise,
                        regret_k=regret_k,
                        remove_frac=(remove_frac_low, remove_frac_high),
                        time_limit=time_limit)
            solucao_inicial, best, best_time = alns.run(seed_init=seed)

            if best is None:
                linha = f"Seed {seed}\nALNS não conseguiu achar uma solução válida.\n"
                f.write(linha)
                print(linha, end="")  # imprime no terminal
                continue

            # primeira linha: seed
            linha = f"{seed}\n"
            f.write(linha); print(linha, end="")
            # segunda linha: solução inicial
            linha = f"{solucao_inicial.C:.2f}\n"
            f.write(linha); print(linha, end="")
            # terceira linha: melhor solução
            linha = f"{best.C:.2f}\n"
            f.write(linha); print(linha, end="")
            # quarta linha: tempo
            linha = f"{best_time:.2f} s\n"
            f.write(linha); print(linha, end="")
            # próximas k linhas: estação, trabalhador, tarefas
            for s_idx in range(data.m):
                w = best.worker_by_station[s_idx]
                tarefas = sorted(best.tasks_by_station[s_idx])
                Ts = best.Ts[s_idx]
                linha = f"Estacao {s_idx+1}, Trabalhador {w}, T_s={Ts:.6f}, Tarefas={tarefas}\n"
                f.write(linha); print(linha, end="")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python script.py <instancia_path> <output_file>")
        sys.exit(1)

    instancia_path = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Processando {instancia_path} -> {output_file}")
    run_alns(instancia_path, output_file=output_file)
