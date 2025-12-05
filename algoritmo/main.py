import os
import random
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
                f.write(f"Seed {seed}\n")
                f.write("ALNS não conseguiu achar uma solução válida.\n")
                continue

            # primeira linha: seed
            f.write(f"{seed}\n")
            # segunda linha: solução inicial
            f.write(f"{solucao_inicial.C:.2f}\n")
            # terceira linha: melhor solução
            f.write(f"{best.C:.2f}\n")
            # quarta linha: tempo
            f.write(f"{best_time:.2f} s\n")
            # próximas k linhas: estação, trabalhador, tarefas
            for s_idx in range(data.m):
                w = best.worker_by_station[s_idx]
                tarefas = sorted(best.tasks_by_station[s_idx])
                Ts = best.Ts[s_idx]
                f.write(f"Estacao {s_idx+1}, Trabalhador {w}, T_s={Ts:.6f}, Tarefas={tarefas}\n")

def run_all_instances(instancias_dir="instancias", resultados_dir="resultados"):
    # cria pasta de resultados se não existir
    os.makedirs(resultados_dir, exist_ok=True)

    # percorre todos os arquivos da pasta instancias
    for filename in os.listdir(instancias_dir):
        instancia_path = os.path.join(instancias_dir, filename)
        if not os.path.isfile(instancia_path):
            continue
        # nome do arquivo de saída na pasta resultados
        output_file = os.path.join(resultados_dir, f"saida_{filename}.txt")
        print(f"Processando {filename} -> {output_file}")
        run_alns(instancia_path, output_file=output_file)

# exemplo de uso
if __name__ == "__main__":
    run_all_instances("instancias", "resultados_heuristica")