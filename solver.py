import sys
# Install gurobipy if not already installed

from gurobipy import Model, GRB, Env, quicksum
import networkx as nx
from algoritmo.ler_instancia import ler_instancia

def resolver_alwabp_gurobi(caminho_instancia, saida_arquivo=None, time_limit=60, seed=0, mip_gap=None):
    # Ler instância
    n, tempos_por_tarefa, precedencias = ler_instancia(caminho_instancia)

    # Determinar k (trabalhadores) e m (estações): versão clássica m = k
    k = len(tempos_por_tarefa[0])
    m = k

    tarefas = list(range(1, n+1))
    trabalhadores = list(range(1, k+1))
    estacoes = list(range(1, m+1))

    # Conjunto de incapacidade I_w: tarefas que o trabalhador w não executa
    incapacidade = {w: set() for w in trabalhadores}
    # tempos t_{w,i}: vou armazenar como dict (w,i) -> t
    t = {} 
    for i in tarefas:
        for w in trabalhadores:
            twi = tempos_por_tarefa[i-1][w-1]
            if twi == float("inf"):
                incapacidade[w].add(i)
            else:
                t[(w, i)] = float(twi)

    # Ambiente e modelo
    env = Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()

    model = Model("ALWABP", env=env)

    # Variáveis:
    # x_{i,s} = 1 se tarefa i é atribuída à estação s
    x = model.addVars(tarefas, estacoes, vtype=GRB.BINARY, name="x")
    # y_{w,s} = 1 se trabalhador w é alocado à estação s
    y = model.addVars(trabalhadores, estacoes, vtype=GRB.BINARY, name="y")
    # z_{w,s,i} = 1 se tarefa i na estação s é feita pelo trabalhador w
    z = model.addVars(trabalhadores, estacoes, tarefas, vtype=GRB.BINARY, name="z")
    # Tempo por estação T_s e tempo de ciclo C
    T = model.addVars(estacoes, vtype=GRB.CONTINUOUS, lb=0.0, name="T")
    C = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="C")

    # Função objetivo: minimizar C
    model.setObjective(C, GRB.MINIMIZE)

    # Restrições

    # 1) Cada tarefa em exatamente uma estação
    for i in tarefas:
        model.addConstr(quicksum(x[i, s] for s in estacoes) == 1, name=f"task_assign_{i}")

    # 2) Cada trabalhador em exatamente uma estação
    for w in trabalhadores:
        model.addConstr(quicksum(y[w, s] for s in estacoes) == 1, name=f"worker_to_one_station_{w}")

    # 3) Cada estação recebe exatamente um trabalhador
    for s in estacoes:
        model.addConstr(quicksum(y[w, s] for w in trabalhadores) == 1, name=f"one_worker_per_station_{s}")

    # 4) Vincular z com x e y: z_{w,s,i} <= x_{i,s} e z_{w,s,i} <= y_{w,s}
    for w in trabalhadores:
        for s in estacoes:
            for i in tarefas:
                model.addConstr(z[w, s, i] <= x[i, s], name=f"z_le_x_w{w}_s{s}_i{i}")
                model.addConstr(z[w, s, i] <= y[w, s], name=f"z_le_y_w{w}_s{s}_i{i}")

    # 5) Para cada par (s,i): soma_w z_{w,s,i} = x_{i,s}
    for s in estacoes:
        for i in tarefas:
            model.addConstr(quicksum(z[w, s, i] for w in trabalhadores) == x[i, s], name=f"link_zx_s{s}_i{i}")

    # 6) Incapacidade: z_{w,s,i} = 0 se i ∈ I_w
    for w in trabalhadores:
        for i in incapacidade[w]:
            for s in estacoes:
                model.addConstr(z[w, s, i] == 0, name=f"incap_w{w}_i{i}_s{s}")

    # 7) Tempo da estação e tempo de ciclo
    for s in estacoes:
        model.addConstr(
            T[s] == quicksum(t[(w, i)] * z[w, s, i] for w in trabalhadores for i in tarefas if (w, i) in t),
            name=f"T_def_{s}"
        )
        model.addConstr(C >= T[s], name=f"C_ge_T_{s}")

    # 8) Precedência: sum_s s*x_{i,s} <= sum_s s*x_{j,s} para (i,j) ∈ E
    for (i, j) in precedencias:
        model.addConstr(
            quicksum(s * x[i, s] for s in estacoes) <= quicksum(s * x[j, s] for s in estacoes),
            name=f"preced_{i}_{j}"
        )

    # Parâmetros do solver
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    if mip_gap is not None:
        model.setParam("MIPGap", mip_gap)
    if seed is not None:
        model.setParam("Seed", seed)

    # Resolver
    model.optimize()

    # Checar solução
    status = model.Status
    if status not in [GRB.OPTIMAL, GRB.INTERRUPTED, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        print(f"Status do solver: {status}. Nenhuma solução viável encontrada.")
        return

    # Extrair solução
    x_val = {(i, s): x[i, s].X for i in tarefas for s in estacoes}
    y_val = {(w, s): y[w, s].X for w in trabalhadores for s in estacoes}
    z_val = {(w, s, i): z[w, s, i].X for w in trabalhadores for s in estacoes for i in tarefas}
    T_val = {s: T[s].X for s in estacoes}
    C_val = C.X

    # Construir solução legível: trabalhador por estação e tarefas por estação
    worker_by_station = {}
    for s in estacoes:
        # O trabalhador que está na estação s (valor próximo de 1)
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

    # Impressão da solução
    print(f"Tempo de ciclo (C): {C_val:.4f}")
    for s in estacoes:
        w = worker_by_station[s]
        print(f"Estação {s}: trabalhador {w}, T_s = {T_val[s]:.4f}, tarefas: {sorted(tasks_by_station[s])}")

    # Gravar solução em arquivo, se solicitado
    if saida_arquivo:
        with open(saida_arquivo, "w") as f:
            f.write(f"Tempo de ciclo C: {C_val:.6f}\n")
            for s in estacoes:
                w = worker_by_station[s]
                tarefas_s = " ".join(map(str, sorted(tasks_by_station[s])))
                f.write(f"Estacao {s}; trabalhador {w}; T_s {T_val[s]:.6f}; tarefas {tarefas_s}\n")


# Call the solver directly with the instance file
resolver_alwabp_gurobi('instancias/1_wee')
