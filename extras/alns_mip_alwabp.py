#!/usr/bin/env python3
# alns_mip_alwabp.py
# ALNS que manipula fixações no modelo MILP (Gurobi) para ALWABP

import sys
import math
import random
import time
from collections import defaultdict
import networkx as nx
from gurobipy import Model, GRB, Env, quicksum

# -------------------------
# I/O e estrutura de dados
# -------------------------

class Instance:
    def __init__(self, path):
        self.n, self.tempos, self.precedencias, self.G = self._read(path)
        self.tarefas = list(range(1, self.n + 1))
        self.k = len(self.tempos[0])
        self.trabalhadores = list(range(1, self.k + 1))
        self.m = self.k  # versão clássica
        self.estacoes = list(range(1, self.m + 1))
        # incapacidade
        self.incapacidade = {w: set() for w in self.trabalhadores}
        for i in self.tarefas:
            for w in self.trabalhadores:
                if self.tempos[i-1][w-1] == float("inf"):
                    self.incapacidade[w].add(i)
        # pred/succ e topo
        self.pred = {i: set(self.G.predecessors(i)) for i in self.tarefas}
        self.succ = {i: set(self.G.successors(i)) for i in self.tarefas}
        self.topo = list(nx.topological_sort(self.G))
        self.topo_index = {t: idx for idx, t in enumerate(self.topo)}

    def _read(self, path):
        with open(path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        n = int(lines[0])
        tempos = []
        idx = 1
        for _ in range(n):
            row = []
            for v in lines[idx].split():
                if v.lower() == "inf":
                    row.append(float("inf"))
                else:
                    row.append(float(v))
            tempos.append(row)
            idx += 1
        preced = []
        while idx < len(lines):
            a, b = lines[idx].split()
            a, b = int(a), int(b)
            if a == -1 and b == -1:
                break
            preced.append((a, b))
            idx += 1
        G = nx.DiGraph()
        G.add_nodes_from(range(1, n+1))
        G.add_edges_from(preced)
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Grafo de precedência contém ciclo.")
        return n, tempos, preced, G

# -------------------------
# Model builder (Gurobi)
# -------------------------

def build_model(instance: Instance, fix_assignments=None, fix_workers=None, time_limit=10, mip_gap=None, seed=0):
    """
    Constrói e retorna um modelo Gurobi para a instância.
    - fix_assignments: dict i -> s (fixa tarefa i na estação s). Para tarefas não presentes, x[i,*] livres.
    - fix_workers: dict s -> w (fixa trabalhador w na estação s). Para estações não presentes, y[*] livres.
    - time_limit: tempo máximo para otimização (segundos).
    - mip_gap: MIPGap (opcional).
    """
    inst = instance
    # The outer try-except for gurobipy import is already handled at the top of the cell.
    # This local check is for redundant safety if the environment changes or if this function is called directly.
    try:
        pass # gurobipy is expected to be imported at the cell level
    except ImportError:
        print("Gurobipy not found in build_model. Ensure it's installed and imported.")
        return None, None, None, None, None

    model = Model("ALWABP_ALNS")
    model.setParam("OutputFlag", 0)
    model.setParam("Seed", seed)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    if mip_gap is not None:
        model.setParam("MIPGap", mip_gap)

    # Variáveis
    x = model.addVars(inst.tarefas, inst.estacoes, vtype=GRB.BINARY, name="x")
    y = model.addVars(inst.trabalhadores, inst.estacoes, vtype=GRB.BINARY, name="y")
    z = model.addVars(inst.trabalhadores, inst.estacoes, inst.tarefas, vtype=GRB.BINARY, name="z")
    T = model.addVars(inst.estacoes, vtype=GRB.CONTINUOUS, lb=0.0, name="T")
    C = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="C")

    # Objetivo
    model.setObjective(C, GRB.MINIMIZE)

    # Restrições
    # 1) cada tarefa em exatamente uma estação
    for i in inst.tarefas:
        model.addConstr(quicksum(x[i, s] for s in inst.estacoes) == 1)

    # 2) cada trabalhador em exatamente uma estação
    for w in inst.trabalhadores:
        model.addConstr(quicksum(y[w, s] for s in inst.estacoes) == 1)

    # 3) cada estação recebe exatamente um trabalhador
    for s in inst.estacoes:
        model.addConstr(quicksum(y[w, s] for w in inst.trabalhadores) == 1)

    # 4) vincular z <= x, z <= y
    for w in inst.trabalhadores:
        for s in inst.estacoes:
            for i in inst.tarefas:
                model.addConstr(z[w, s, i] <= x[i, s])
                model.addConstr(z[w, s, i] <= y[w, s])

    # 5) sum_w z = x
    for s in inst.estacoes:
        for i in inst.tarefas:
            model.addConstr(quicksum(z[w, s, i] for w in inst.trabalhadores) == x[i, s])

    # 6) incapacidade
    for w in inst.trabalhadores:
        for i in inst.incapacidade[w]:
            for s in inst.estacoes:
                model.addConstr(z[w, s, i] == 0)

    # 7) T_s definition and C >= T_s
    for s in inst.estacoes:
        model.addConstr(T[s] == quicksum(inst.tempos[i-1][w-1] * z[w, s, i]
                                         for w in inst.trabalhadores for i in inst.tarefas if inst.tempos[i-1][w-1] != float("inf")))
        model.addConstr(C >= T[s])

    # 8) precedência: sum s*x_{i,s} <= sum s*x_{j,s}
    for (i, j) in inst.precedencias:
        model.addConstr(quicksum(s * x[i, s] for s in inst.estacoes) <= quicksum(s * x[j, s] for s in inst.estacoes))

    # Aplicar fixações (fix_assignments e fix_workers)
    if fix_assignments:
        for i, s_fixed in fix_assignments.items():
            # fixa x[i, s_fixed] = 1 e outros x[i, s'] = 0
            for s in inst.estacoes:
                if s == s_fixed:
                    model.addConstr(x[i, s] == 1)
                else:
                    model.addConstr(x[i, s] == 0)

    if fix_workers:
        for s, w_fixed in fix_workers.items():
            for w in inst.trabalhadores:
                if w == w_fixed:
                    model.addConstr(y[w, s] == 1)
                else:
                    model.addConstr(y[w, s] == 0)

    model.update()
    return model, x, y, z, T, C

# -------------------------
# Heurísticas (escolha de tarefas para "remover")
# -------------------------

def select_random_remove(current_assign, q, rng):
    """current_assign: dict i->s (solução atual). Retorna lista de tarefas a remover."""
    tasks = list(current_assign.keys())
    rng.shuffle(tasks)
    return tasks[:q]

def select_worst_remove(instance: Instance, current_assign, q):
    """Remove tarefas com maior contribuição (tempo no trabalhador atual)."""
    contrib = []
    for i, s in current_assign.items():
        # trabalhador w at station s: we need worker_by_station mapping
        # current_assign must be accompanied by worker_by_station mapping externally
        pass  # implemented in wrapper below

def select_shaw_remove(instance: Instance, current_assign, worker_by_station, q, rng):
    """Shaw-like: pick seed and remove tasks similar by topo distance and time."""
    tasks = list(current_assign.keys())
    if not tasks:
        return []
    seed = rng.choice(tasks)
    def related(a, b):
        da = instance.topo_index[a]
        db = instance.topo_index[b]
        dtopo = abs(da - db)
        # time at assigned worker
        wa = worker_by_station[current_assign[a]-1]
        wb = worker_by_station[current_assign[b]-1]
        ta = instance.tempos[a-1][wa-1] if instance.tempos[a-1][wa-1] != float("inf") else 1e6
        tb = instance.tempos[b-1][wb-1] if instance.tempos[b-1][wb-1] != float("inf") else 1e6
        dtime = abs(ta - tb) / max(1.0, (ta + tb))
        return dtopo + dtime
    ranked = sorted(tasks, key=lambda t: related(seed, t))
    return ranked[:q]

# Wrapper for worst remove (needs worker_by_station)
def select_worst_remove_wrapper(instance: Instance, current_assign, worker_by_station, q):
    contrib = []
    for i, s in current_assign.items():
        w = worker_by_station[s-1]
        t = instance.tempos[i-1][w-1]
        contrib.append((t, i))
    contrib.sort(reverse=True)
    return [i for _, i in contrib[:q]]

# -------------------------
# Utilitários para extrair solução do modelo
# -------------------------

def extract_solution_from_model(model, x, y, inst: Instance):
    """Retorna (assignments dict i->s, worker_by_station list indexed 1..m, C value)."""
    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.INTERRUPTED):
        return None
    assign = {} 
    # Check if model has a solution before trying to access .X attribute
    if model.solCount > 0:
        for i in inst.tarefas:
            for s in inst.estacoes:
                if x[i, s].X > 0.5:
                    assign[i] = s
                    break
        worker_by_station = [None] * inst.m
        for s in inst.estacoes:
            for w in inst.trabalhadores:
                if y[w, s].X > 0.5:
                    worker_by_station[s-1] = w
                    break
        Cval = None
        try:
            Cvar = [v for v in model.getVars() if v.VarName == "C"][0]
            Cval = Cvar.X
        except Exception:
            # fallback: compute from T variables if available
            Cval = None # This might need a proper recalculation if Cvar is not found/accessible
        return assign, worker_by_station, Cval
    else:
        return None, None, None # No solution found

# -------------------------
# ALNS sobre MILP
# -------------------------

class ALNS_MIP:
    def __init__(self, instance: Instance, seed=0,
                 time_limit_per_reopt=5.0, mip_gap=None,
                 max_iter=200, segment_len=50,
                 sigma1=10.0, sigma2=5.0, sigma3=1.0, r=0.2,
                 remove_frac=(0.2, 0.4)):
        self.inst = instance
        self.rng = random.Random(seed)
        self.time_limit_per_reopt = time_limit_per_reopt
        self.mip_gap = mip_gap
        self.max_iter = max_iter
        self.segment_len = segment_len
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.r = r
        self.remove_frac = remove_frac

        # heuristics lists
        self.remove_ops = [
            ("random", lambda cur, wbs, q: select_random_remove(cur, q, self.rng)),
            ("worst", lambda cur, wbs, q: select_worst_remove_wrapper(self.inst, cur, wbs, q)),
            ("shaw", lambda cur, wbs, q: select_shaw_remove(self.inst, cur, wbs, q, self.rng)),
        ]
        self.remove_weights = [1.0] * len(self.remove_ops)
        self.remove_scores = [0.0] * len(self.remove_ops)
        self.remove_uses = [0] * len(self.remove_ops)

    def roulette(self, weights):
        total = sum(weights)
        r = self.rng.random() * total
        acc = 0.0
        for idx, w in enumerate(weights):
            acc += w
            if r <= acc:
                return idx
        return len(weights) - 1

    def update_weights(self):
        for i in range(len(self.remove_weights)):
            if self.remove_uses[i] > 0:
                self.remove_weights[i] = (1 - self.r) * self.remove_weights[i] + self.r * (self.remove_scores[i] / self.remove_uses[i])
            self.remove_scores[i] = 0.0
            self.remove_uses[i] = 0

    def initial_solution_greedy(self, seed=0):
        """Gera solução inicial gulosa (workers shuffled + topological greedy)."""
        rng = random.Random(seed)
        # assign workers identity-shuffled
        workers = list(self.inst.trabalhadores)
        rng.shuffle(workers)
        worker_by_station = workers[:]  # station s has worker workers[s-1]
        assign = {}
        Ts = [0.0] * self.inst.m
        # Initialize C to a reasonable starting value, e.g., 0.0 or a very small number
        C = 0.0
        for i in self.inst.topo:
            best_s = None
            best_Cproxy = float("inf")
            for s in self.inst.estacoes:
                # check precedence feasibility quickly using assign
                ok = True
                for p in self.inst.pred[i]:
                    if p in assign and assign[p] > s:
                        ok = False
                        break
                if not ok:
                    continue
                w = worker_by_station[s-1]
                if i in self.inst.incapacidade[w]:
                    continue
                new_load = Ts[s-1] + self.inst.tempos[i-1][w-1]
                proxyC = max(new_load, C) # Use current C value
                if proxyC < best_Cproxy:
                    best_Cproxy = proxyC
                    best_s = s
            if best_s is None:
                return None, None, None # Failed to assign a task, initial solution is not feasible
            assign[i] = best_s
            Ts[best_s-1] += self.inst.tempos[i-1][worker_by_station[best_s-1]-1]
            C = max(Ts) # Update C after task assignment
        return assign, worker_by_station, C

    def run(self, seed_init=0):
        # inicial
        cur_assign, cur_workers, cur_C = self.initial_solution_greedy(seed=seed_init)
        if cur_assign is None:
            print("Failed to generate greedy initial solution, attempting full MILP fallback.")
            # fallback: solve full MILP once (mais caro) to get feasible
            # The gurobipy import is now handled at the top of the cell. No need for local import.

            model, x, y, z, T, Cvar = build_model(self.inst, time_limit=60, mip_gap=self.mip_gap, seed=seed_init)
            if model is None: # Check if build_model failed
                raise RuntimeError("Failed to build Gurobi model for initial solution.")

            model.optimize()
            sol = extract_solution_from_model(model, x, y, self.inst)
            if sol[0] is None: # Check sol[0] for assignment dictionary
                raise RuntimeError("Não foi possível obter solução inicial viável com o fallback MILP.")
            cur_assign, cur_workers, cur_C = sol[0], sol[1], sol[2]

        best_assign = dict(cur_assign)
        best_workers = list(cur_workers)
        best_C = cur_C

        iter_count = 0
        segment_counter = 0

        while iter_count < self.max_iter:
            # escolher heurística de remoção
            ridx = self.roulette(self.remove_weights)
            rname, rop = self.remove_ops[ridx]
            self.remove_uses[ridx] += 1

            # definir q
            qmin = max(1, int(self.remove_frac[0] * self.inst.n))
            qmax = max(qmin, int(self.remove_frac[1] * self.inst.n))
            q = self.rng.randint(qmin, qmax)

            # escolher tarefas a remover (liberar no modelo)
            to_remove = rop(cur_assign, cur_workers, q)

            # construir fix_assignments: fixa todas as tarefas exceto as removidas
            fix_assign = {i: s for i, s in cur_assign.items() if i not in to_remove}
            # opcional: permitir troca de alguns workers (aqui mantemos workers fixos para simplicidade)
            fix_workers = {s+1: cur_workers[s] for s in range(len(cur_workers))}  # fixa todos os workers
            # se quiser permitir troca de workers em algumas iterações, remova algumas entradas de fix_workers

            # construir e resolver modelo restrito
            model, x, y, z, Tvar, Cvar = build_model(self.inst, fix_assignments=fix_assign,
                                                     fix_workers=fix_workers,
                                                     time_limit=self.time_limit_per_reopt,
                                                     mip_gap=self.mip_gap,
                                                     seed=self.rng.randint(1, 10**6))
            if model is None: # Check if build_model failed
                self.remove_scores[ridx] += 0.0 # Penalize lightly
                iter_count += 1
                segment_counter += 1
                if segment_counter >= self.segment_len:
                    self.update_weights()
                    segment_counter = 0
                continue

            model.optimize()

            new_assign, new_workers, new_C = extract_solution_from_model(model, x, y, self.inst)
            if new_assign is None: # Check new_assign to indicate if a solution was found
                # falha em resolver; penaliza heurística levemente e continua
                self.remove_scores[ridx] += 0.0
                iter_count += 1
                segment_counter += 1
                if segment_counter >= self.segment_len:
                    self.update_weights()
                    segment_counter = 0
                continue

            # critério de aceitação (Simulated Annealing simples)
            accepted = False
            if new_C < cur_C: # new_C is guaranteed not None if new_assign is not None
                accepted = True
            else:
                # aceita pior com prob pequena (temperatura fixa pequena)
                T = max(1.0, 0.1 * cur_C) # Use cur_C as reference for temperature scaling
                prob = math.exp(-(new_C - cur_C) / T)
                if self.rng.random() < prob:
                    accepted = True

            # atualizar registros e pesos
            if new_C < best_C: # new_C is guaranteed not None
                best_assign = dict(new_assign)
                best_workers = list(new_workers)
                best_C = new_C
                self.remove_scores[ridx] += self.sigma1
            elif accepted and new_C < cur_C: # new_C is guaranteed not None
                self.remove_scores[ridx] += self.sigma2
            elif accepted: # Accepted a worse solution (new_C >= cur_C)
                self.remove_scores[ridx] += self.sigma3

            if accepted:
                cur_assign = dict(new_assign)
                cur_workers = list(new_workers)
                cur_C = new_C

            iter_count += 1
            segment_counter += 1
            if segment_counter >= self.segment_len:
                self.update_weights()
                segment_counter = 0

        return best_assign, best_workers, best_C

# -------------------------
# CLI / execução
# -------------------------

def main():
    # Correct way to instantiate ALNS_MIP with an Instance object
    instance_path = "instancias/72_wee" # Changed from "instancias/1_hes" to "17_hes"
    # Create an Instance object first
    instance_data = Instance(instance_path)
    # Pass the Instance object to ALNS_MIP
    alns = ALNS_MIP(instance_data)
    best_assign, best_workers, best_C = alns.run()

    print("Melhor C encontrado:", best_C)
    print("Workers por estação:")
    for s, w in enumerate(best_workers, start=1):
        print(f"  Estação {s}: trabalhador {w}")
    print("Atribuições (tarefa -> estação):")
    for i in sorted(best_assign.keys()):
        print(f"  Tarefa {i} -> Estação {best_assign[i]}")

if __name__ == "__main__":
    main()