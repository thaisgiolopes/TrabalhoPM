import sys
import time
from gurobipy import Model, GRB, Env, quicksum
import networkx as nx
from algoritmo.ler_instancia import ler_instancia
import os


def resolver_alwabp_gurobi(caminho_instancia, saida_arquivo=None, time_limit=1800, seed=0, mip_gap=None):
    # Medir tempo de execução
    inicio = time.time()

    # Ler instância
    n, tempos_por_tarefa, precedencias = ler_instancia(caminho_instancia)

    k = len(tempos_por_tarefa[0])
    m = k

    tarefas = list(range(1, n+1))
    trabalhadores = list(range(1, k+1))
    estacoes = list(range(1, m+1))

    incapacidade = {w: set() for w in trabalhadores}
    t = {}
    for i in tarefas:
        for w in trabalhadores:
            twi = tempos_por_tarefa[i-1][w-1]
            if twi == float("inf"):
                incapacidade[w].add(i)
            else:
                t[(w, i)] = float(twi)

    env = Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()

    model = Model("ALWABP", env=env)

    x = model.addVars(tarefas, estacoes, vtype=GRB.BINARY, name="x")
    y = model.addVars(trabalhadores, estacoes, vtype=GRB.BINARY, name="y")
    z = model.addVars(trabalhadores, estacoes, tarefas, vtype=GRB.BINARY, name="z")
    T = model.addVars(estacoes, vtype=GRB.CONTINUOUS, lb=0.0, name="T")
    C = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="C")

    model.setObjective(C, GRB.MINIMIZE)

    for i in tarefas:
        model.addConstr(quicksum(x[i, s] for s in estacoes) == 1)

    for w in trabalhadores:
        model.addConstr(quicksum(y[w, s] for s in estacoes) == 1)

    for s in estacoes:
        model.addConstr(quicksum(y[w, s] for w in trabalhadores) == 1)

    for w in trabalhadores:
        for s in estacoes:
            for i in tarefas:
                model.addConstr(z[w, s, i] <= x[i, s])
                model.addConstr(z[w, s, i] <= y[w, s])

    for s in estacoes:
        for i in tarefas:
            model.addConstr(quicksum(z[w, s, i] for w in trabalhadores) == x[i, s])

    for w in trabalhadores:
        for i in incapacidade[w]:
            for s in estacoes:
                model.addConstr(z[w, s, i] == 0)

    for s in estacoes:
        model.addConstr(
            T[s] == quicksum(t[(w, i)] * z[w, s, i] for w in trabalhadores for i in tarefas if (w, i) in t)
        )
        model.addConstr(C >= T[s])

    for (i, j) in precedencias:
        model.addConstr(
            quicksum(s * x[i, s] for s in estacoes) <= quicksum(s * x[j, s] for s in estacoes)
        )

    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    if mip_gap is not None:
        model.setParam("MIPGap", mip_gap)
    if seed is not None:
        model.setParam("Seed", seed)

    model.optimize()

    fim = time.time()
    tempo_execucao = fim - inicio

    status = model.Status
    if status not in [GRB.OPTIMAL, GRB.INTERRUPTED, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        print(f"Status do solver: {status}. Nenhuma solução viável encontrada.")
        return

    x_val = {(i, s): x[i, s].X for i in tarefas for s in estacoes}
    y_val = {(w, s): y[w, s].X for w in trabalhadores for s in estacoes}
    T_val = {s: T[s].X for s in estacoes}
    C_val = C.X

    worker_by_station = {}
    for s in estacoes:
        w_assigned = None
        for w in trabalhadores:
            if y_val[(w, s)] > 0.5:
                w_assigned = w
                break
        worker_by_station[s] = w_assigned

    tasks_by_station = {s: [] for s in estacoes}
    for s in estacoes:
        for i in tarefas:
            if x_val[(i, s)] > 0.5:
                tasks_by_station[s].append(i)

    print(f"Tempo de ciclo (C): {C_val:.4f}")
    print(f"Tempo de execução: {tempo_execucao:.2f} segundos")
    for s in estacoes:
        w = worker_by_station[s]
        print(f"Estação {s}: trabalhador {w}, T_s = {T_val[s]:.4f}, tarefas: {sorted(tasks_by_station[s])}")

    if saida_arquivo:
        os.makedirs(os.path.dirname(saida_arquivo), exist_ok=True)
        with open(saida_arquivo, "w") as f:
            f.write(f"Tempo de ciclo C: {C_val:.6f}\n")
            f.write(f"Tempo de execução: {tempo_execucao:.2f} segundos\n")
            for s in estacoes:
                w = worker_by_station[s]
                tarefas_s = " ".join(map(str, sorted(tasks_by_station[s])))
                f.write(f"Estacao {s}; trabalhador {w}; T_s {T_val[s]:.6f}; tarefas {tarefas_s}\n")


# Executar solver para todas as instâncias no diretório 'instancias'
os.makedirs('resultados', exist_ok=True)

for arquivo in os.listdir('instancias'):
    caminho = os.path.join('instancias', arquivo)
    print(f"\n=== Resolvendo instância: {caminho} ===")

    nome_instancia = os.path.splitext(arquivo)[0]
    saida_arquivo = os.path.join('resultados', f"{nome_instancia}_resultado.txt")

    resolver_alwabp_gurobi(caminho, saida_arquivo=saida_arquivo)
