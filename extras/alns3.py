#!/usr/bin/env python3
# alns_puro_alwabp.py
# ALNS heurístico para ALWABP sem solver e sem NetworkX

import sys
import math
import random
import time
from collections import deque, defaultdict
from ler_instancia import ler_instancia

# -------------------------------------------------
# Módulo: leitura e dados do problema (sem grafo)
# -------------------------------------------------

class Instance:
    def __init__(self, path):
        self.n, self.tempos, self.precedencias = ler_instancia(path)
        self.tarefas = list(range(1, self.n + 1))
        self.k = len(self.tempos[0])
        self.trabalhadores = list(range(1, self.k + 1))
        self.m = self.k  # versão clássica: |S|=|W|
        self.estacoes = list(range(1, self.m + 1))

        # Incapacidade por trabalhador
        self.incapacidade = {w: set() for w in self.trabalhadores}
        for i in self.tarefas:
            for w in self.trabalhadores:
                if self.tempos[i-1][w-1] == float("inf"):
                    self.incapacidade[w].add(i)

        # Pred/succ como conjuntos
        self.pred = {i: set() for i in self.tarefas}
        self.succ = {i: set() for i in self.tarefas}
        for (i, j) in self.precedencias:
            self.succ[i].add(j)
            self.pred[j].add(i)

        # Ordem topológica por Kahn (sem NetworkX)
        self.topo_order, self.topo_index = self._topological_sort()

    def _topological_sort(self):
        indeg = {i: 0 for i in self.tarefas}
        for (i, j) in self.precedencias:
            indeg[j] += 1
        q = deque([i for i in self.tarefas if indeg[i] == 0])
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in self.succ[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(order) != self.n:
            raise ValueError("Grafo de precedência possui ciclo (instância inválida).")
        topo_index = {i: idx for idx, i in enumerate(order)}
        return order, topo_index

    def twi(self, w, i):
        return self.tempos[i-1][w-1]

# -------------------------------------------------
# Módulo: solução (variáveis x, y implícitas)
# -------------------------------------------------

class Solution:
    def __init__(self, inst: Instance):
        self.inst = inst
        # y[s] = w (1..m -> 1..k)
        self.worker_by_station = [None for _ in inst.estacoes]
        # x[i] = s (atribuição de tarefa à estação)
        self.task_to_station = {i: None for i in inst.tarefas}
        # tarefas por estação (listas)
        self.tasks_by_station = [[] for _ in inst.estacoes]
        # cargas por estação T_s e tempo de ciclo C
        self.Ts = [0.0 for _ in inst.estacoes]
        self.C = float("inf")

    def clone(self):
        s2 = Solution(self.inst)
        s2.worker_by_station = list(self.worker_by_station)
        s2.task_to_station = dict(self.task_to_station)
        s2.tasks_by_station = [list(lst) for lst in self.tasks_by_station]
        s2.Ts = list(self.Ts)
        s2.C = self.C
        return s2

    def set_workers(self, workers):
        """Define trabalhadores por estação e recalcula Ts."""
        self.worker_by_station = list(workers)
        self._recompute_Ts()

    def _recompute_Ts(self):
        """Recalcula cargas por estação e C."""
        inst = self.inst
        self.Ts = [0.0 for _ in inst.estacoes]
        infeasible = False
        for s_idx in range(inst.m):
            w = self.worker_by_station[s_idx]
            total = 0.0
            for i in self.tasks_by_station[s_idx]:
                t = inst.twi(w, i)
                if t == float("inf"):
                    infeasible = True
                    total = float("inf")
                    break
                total += t
            self.Ts[s_idx] = total
        self.C = max(self.Ts) if self.Ts else (float("inf") if infeasible else 0.0)

    def insert_task(self, i, station_idx):
        """Insere tarefa i na estação station_idx (1..m) e atualiza incrementalmente."""
        inst = self.inst
        s = station_idx - 1
        w = self.worker_by_station[s]
        t = inst.twi(w, i)
        if t == float("inf"):
            return False
        # Inserção
        self.tasks_by_station[s].append(i)
        self.task_to_station[i] = station_idx
        # Atualização incremental
        self.Ts[s] += t
        self.C = max(self.Ts)
        return True

    def remove_task(self, i):
        """Remove tarefa i de sua estação atual e atualiza incrementalmente."""
        inst = self.inst
        s_idx = self.task_to_station[i]
        if s_idx is None:
            return False
        s = s_idx - 1
        w = self.worker_by_station[s]
        t = inst.twi(w, i)
        if i in self.tasks_by_station[s]:
            self.tasks_by_station[s].remove(i)
        self.task_to_station[i] = None
        if t != float("inf"):
            self.Ts[s] -= t
        self.C = max(self.Ts)
        return True

# -------------------------------------------------
# Módulo: viabilidade e construção inicial
# -------------------------------------------------

def precedence_ok_on_station(sol: Solution, task, station_idx):
    """Verifica i ⪯ j via estações: pred(i) em <= station_idx e succ(i) em >= station_idx."""
    inst = sol.inst
    for p in inst.pred[task]:
        sp = sol.task_to_station[p]
        if sp is not None and sp > station_idx:
            return False
    for s in inst.succ[task]:
        ss = sol.task_to_station[s]
        if ss is not None and station_idx > ss:
            return False
    return True

def initial_solution_greedy(inst: Instance, seed=0):
    rng = random.Random(seed)
    sol = Solution(inst)
    workers = list(inst.trabalhadores)
    rng.shuffle(workers)
    sol.set_workers(workers)

    # Inserção na ordem topológica
    for i in inst.topo_order:
        best_s, best_cost = None, float("inf")
        for station_idx in inst.estacoes:
            if not precedence_ok_on_station(sol, i, station_idx):
                continue
            w = sol.worker_by_station[station_idx - 1]
            if i in inst.incapacidade[w]:
                continue
            t = inst.twi(w, i)
            new_load = sol.Ts[station_idx - 1] + t
            proxy_C = max(new_load, sol.C)
            if proxy_C < best_cost:
                best_cost, best_s = proxy_C, station_idx
        if best_s is None:
            return None
        sol.insert_task(i, best_s)
    return sol

# -------------------------------------------------
# Módulo: operadores de remoção
# -------------------------------------------------

def remove_random(sol: Solution, q, rng):
    removed = []
    all_tasks = [i for lst in sol.tasks_by_station for i in lst]
    rng.shuffle(all_tasks)
    for i in all_tasks[:q]:
        sol.remove_task(i)
        removed.append(i)
    return removed

def remove_worst(sol: Solution, q, rng):
    """Remove tarefas de maior tempo na estação atual (aproxima contribuição)."""
    inst = sol.inst
    contrib = []
    for s_idx in range(inst.m):
        w = sol.worker_by_station[s_idx]
        for i in sol.tasks_by_station[s_idx]:
            contrib.append((inst.twi(w, i), i))
    contrib.sort(reverse=True)
    removed = []
    for _, i in contrib[:q]:
        sol.remove_task(i)
        removed.append(i)
    return removed

def remove_shaw(sol: Solution, q, rng):
    """Shaw Removal leve: proximidade por topo_index e tempos em workers atuais."""
    inst = sol.inst
    all_tasks = [i for lst in sol.tasks_by_station for i in lst]
    if not all_tasks:
        return []
    seed_task = rng.choice(all_tasks)
    def related(a, b):
        da = inst.topo_index[a]
        db = inst.topo_index[b]
        dtopo = abs(da - db)
        wa = sol.worker_by_station[sol.task_to_station[a]-1]
        wb = sol.worker_by_station[sol.task_to_station[b]-1]
        ta = inst.twi(wa, a)
        tb = inst.twi(wb, b)
        dtime = abs(ta - tb) / max(1.0, (ta + tb))
        return dtopo + dtime
    ranked = sorted(all_tasks, key=lambda t: related(seed_task, t))
    removed = []
    for i in ranked[:q]:
        sol.remove_task(i)
        removed.append(i)
    return removed

# -------------------------------------------------
# Módulo: inserção (greedy e regret-k)
# -------------------------------------------------

def candidate_stations_limited(sol: Solution, task):
    """Estações candidatas: atual, anterior e seguinte; se não alocado, janela por pred/succ com expansão ±1."""
    inst = sol.inst
    s_cur = sol.task_to_station[task]
    cand = set()
    if s_cur is not None:
        for s in [s_cur - 1, s_cur, s_cur + 1]:
            if 1 <= s <= inst.m:
                cand.add(s)
    else:
        min_pos = 1
        max_pos = inst.m
        preds_pos = [sol.task_to_station[p] for p in inst.pred[task] if sol.task_to_station[p] is not None]
        succs_pos = [sol.task_to_station[s] for s in inst.succ[task] if sol.task_to_station[s] is not None]
        if preds_pos:
            min_pos = max(min_pos, max(preds_pos))
        if succs_pos:
            max_pos = min(max_pos, min(succs_pos))
        for s in range(max(1, min_pos - 1), min(inst.m, max_pos + 1) + 1):
            cand.add(s)
    return sorted(cand) if cand else list(inst.estacoes)

def insert_greedy(sol: Solution, tasks_to_insert, rng):
    inst = sol.inst
    for i in tasks_to_insert:
        best_s, best_cost = None, float("inf")
        for station_idx in candidate_stations_limited(sol, i):
            if not precedence_ok_on_station(sol, i, station_idx):
                continue
            w = sol.worker_by_station[station_idx - 1]
            if i in inst.incapacidade[w]:
                continue
            t = inst.twi(w, i)
            new_load = sol.Ts[station_idx - 1] + t
            proxy_C = max(new_load, sol.C)
            if proxy_C < best_cost:
                best_cost, best_s = proxy_C, station_idx
        if best_s is None:
            return False
        sol.insert_task(i, best_s)
    return True

def insert_regret(sol: Solution, tasks_to_insert, rng, k=2):
    inst = sol.inst
    pending = set(tasks_to_insert)
    while pending:
        best_opts = {}
        for i in list(pending):
            opts = []
            for station_idx in candidate_stations_limited(sol, i):
                if not precedence_ok_on_station(sol, i, station_idx):
                    continue
                w = sol.worker_by_station[station_idx - 1]
                if i in inst.incapacidade[w]:
                    continue
                t = inst.twi(w, i)
                new_load = sol.Ts[station_idx - 1] + t
                proxy_C = max(new_load, sol.C)
                opts.append((proxy_C, station_idx))
            opts.sort()
            if opts:
                best_opts[i] = opts[:max(k, 1)]
        if not best_opts:
            return False
        chosen_task, chosen_station, best_regret = None, None, -float("inf")
        for i, opts in best_opts.items():
            if len(opts) == 1:
                regret = float("inf")
            else:
                regret = opts[k-1][0] - opts[0][0]
            if regret > best_regret:
                best_regret = regret
                chosen_task = i
                chosen_station = opts[0][1]
        sol.insert_task(chosen_task, chosen_station)
        pending.remove(chosen_task)
    return True

# -------------------------------------------------
# Módulo: operadores sobre trabalhadores
# -------------------------------------------------

def swap_workers_local(sol: Solution, rng):
    """Troca dois trabalhadores e recalcula Ts apenas dessas estações."""
    inst = sol.inst
    a, b = rng.sample(range(inst.m), 2)
    sol.worker_by_station[a], sol.worker_by_station[b] = sol.worker_by_station[b], sol.worker_by_station[a]
    # Recalcular cargas localmente
    for s in (a, b):
        w = sol.worker_by_station[s]
        total = 0.0
        bad = False
        for i in sol.tasks_by_station[s]:
            t = inst.twi(w, i)
            if t == float("inf"):
                bad = True
                break
            total += t
        sol.Ts[s] = float("inf") if bad else total
    sol.C = max(sol.Ts)
    return True

def reassign_worst_station_worker(sol: Solution, rng):
    """Troca o trabalhador da estação gargalo com uma estação aleatória."""
    worst_idx = max(range(len(sol.Ts)), key=lambda s: sol.Ts[s])
    other = rng.choice([i for i in range(len(sol.Ts)) if i != worst_idx])
    sol.worker_by_station[worst_idx], sol.worker_by_station[other] = sol.worker_by_station[other], sol.worker_by_station[worst_idx]
    # Recalcular duas estações
    for s in (worst_idx, other):
        w = sol.worker_by_station[s]
        total = 0.0
        bad = False
        for i in sol.tasks_by_station[s]:
            t = sol.inst.twi(w, i)
            if t == float("inf"):
                bad = True
                break
            total += t
        sol.Ts[s] = float("inf") if bad else total
    sol.C = max(sol.Ts)
    return True

# -------------------------------------------------
# Módulo: aceitação e ALNS
# -------------------------------------------------

def accept(old_cost, new_cost, T, rng):
    if new_cost < old_cost:
        return True
    delta = new_cost - old_cost
    prob = math.exp(-delta / max(T, 1e-9))
    return rng.random() < prob

class ALNS:
    def __init__(self, inst: Instance, seed=0, segment_length=100, r=0.2,
                 sigma1=10.0, sigma2=5.0, sigma3=1.0, T_start=1.0, cooling=0.995,
                 max_iter=5000, remove_frac=(0.2, 0.4), regret_k=2, time_limit=None, patience=500):
        self.inst = inst
        self.rng = random.Random(seed)
        self.segment_length = segment_length
        self.r = r
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.cooling = cooling
        self.max_iter = max_iter
        self.remove_frac = remove_frac
        self.regret_k = regret_k
        self.time_limit = time_limit
        self.patience = patience
        self.T = T_start

        self.remove_ops = [
            ("random_remove", remove_random),
            ("worst_remove", remove_worst),
            ("shaw_remove", remove_shaw),
        ]
        self.insert_ops = [
            ("greedy_insert", insert_greedy),
            ("regret_insert", lambda sol, tasks, rng: insert_regret(sol, tasks, rng, k=self.regret_k)),
        ]
        self.worker_ops = [
            ("swap_workers_local", swap_workers_local),
            ("reassign_worst_station_worker", reassign_worst_station_worker),
        ]

        self.remove_weights = [1.0] * len(self.remove_ops)
        self.insert_weights = [1.0] * len(self.insert_ops)
        self.worker_weights = [1.0] * len(self.worker_ops)
        self.remove_scores = [0.0] * len(self.remove_ops)
        self.insert_scores = [0.0] * len(self.insert_ops)
        self.worker_scores = [0.0] * len(self.worker_ops)
        self.remove_uses = [0] * len(self.remove_ops)
        self.insert_uses = [0] * len(self.insert_ops)
        self.worker_uses = [0] * len(self.worker_ops)

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
        for arr_w, arr_sc, arr_use in [
            (self.remove_weights, self.remove_scores, self.remove_uses),
            (self.insert_weights, self.insert_scores, self.insert_uses),
            (self.worker_weights, self.worker_scores, self.worker_uses),
        ]:
            for i in range(len(arr_w)):
                if arr_use[i] > 0:
                    arr_w[i] = (1 - self.r) * arr_w[i] + self.r * (arr_sc[i] / arr_use[i])
                arr_sc[i] = 0.0
                arr_use[i] = 0

    def run(self, seed_init=0):
        start_time = time.time()

        current = initial_solution_greedy(self.inst, seed=seed_init)
        if current is None:
            raise RuntimeError("Falha ao construir solução inicial viável.")

        best = current.clone()
        best_cost = best.C
        current_cost = current.C

        visited = set()
        def signature(sol: Solution):
            return tuple((sol.worker_by_station[s], tuple(sorted(sol.tasks_by_station[s]))) for s in range(self.inst.m))
        visited.add(signature(current))

        iter_count = 0
        segment_counter = 0
        no_improve = 0

        while iter_count < self.max_iter:
            if self.time_limit and (time.time() - start_time) >= self.time_limit:
                break
            if no_improve >= self.patience:
                break

            # Seleção por roleta
            ridx = self.roulette(self.remove_weights)
            iidx = self.roulette(self.insert_weights)
            widx = self.roulette(self.worker_weights)

            r_name, r_op = self.remove_ops[ridx]
            i_name, i_op = self.insert_ops[iidx]
            w_name, w_op = self.worker_ops[widx]

            self.remove_uses[ridx] += 1
            self.insert_uses[iidx] += 1
            self.worker_uses[widx] += 1

            candidate = current.clone()

            # Perturbação de trabalhadores com probabilidade moderada
            if self.rng.random() < 0.3:
                w_op(candidate, self.rng)

            # Ruin/Recreate
            q_min = max(4, int(self.remove_frac[0] * self.inst.n))
            q_max = min(100, int(self.remove_frac[1] * self.inst.n))
            q = self.rng.randint(q_min, max(q_min, q_max))

            removed_tasks = r_op(candidate, q, self.rng)
            ok = i_op(candidate, removed_tasks, self.rng)
            if not ok:
                # fallback guloso
                ok = insert_greedy(candidate, removed_tasks, self.rng)
                if not ok:
                    # não conseguiu reconstruir — descarta
                    self.T *= self.cooling
                    iter_count += 1
                    segment_counter += 1
                    if segment_counter >= self.segment_length:
                        self.update_weights()
                        segment_counter = 0
                    no_improve += 1
                    continue

            new_cost = candidate.C
            is_new_best = new_cost < best_cost
            is_better_than_current = new_cost < current_cost
            sig = signature(candidate)
            is_new_solution = sig not in visited

            accepted = accept(current_cost, new_cost, self.T, self.rng)
            if accepted:
                current = candidate
                current_cost = new_cost

            if is_new_best:
                best = candidate.clone()
                best_cost = new_cost
                no_improve = 0
                self.remove_scores[ridx] += self.sigma1
                self.insert_scores[iidx] += self.sigma1
                self.worker_scores[widx] += self.sigma1
            elif is_better_than_current and is_new_solution:
                no_improve = 0
                self.remove_scores[ridx] += self.sigma2
                self.insert_scores[iidx] += self.sigma2
                self.worker_scores[widx] += self.sigma2
            else:
                no_improve += 1
                if accepted and is_new_solution:
                    self.remove_scores[ridx] += self.sigma3
                    self.insert_scores[iidx] += self.sigma3
                    self.worker_scores[widx] += self.sigma3

            if is_new_solution:
                visited.add(sig)

            # resfriamento e pesos
            self.T *= self.cooling
            iter_count += 1
            segment_counter += 1
            if segment_counter >= self.segment_length:
                self.update_weights()
                segment_counter = 0

        return best

# -------------------------------------------------
# Módulo: execução (CLI)
# -------------------------------------------------

def run_alns(instancia_path, seed=42, time_limit=None, max_iter=5000, segment_length=100,
             cooling=0.995, regret_k=2, remove_low=0.2, remove_high=0.4, patience=500):
    inst = Instance(instancia_path)
    alns = ALNS(inst,
                seed=seed,
                segment_length=segment_length,
                cooling=cooling,
                max_iter=max_iter,
                regret_k=regret_k,
                remove_frac=(remove_low, remove_high),
                time_limit=time_limit,
                patience=patience)
    best = alns.run(seed_init=seed)

    print(f"Tempo de ciclo (ALNS) C: {best.C:.6f}")
    for s_idx in range(inst.m):
        w = best.worker_by_station[s_idx]
        tarefas = sorted(best.tasks_by_station[s_idx])
        Ts = best.Ts[s_idx]
        print(f"Estação {s_idx+1}: trabalhador {w}, T_s = {Ts:.6f}, tarefas: {tarefas}")

    return best

if __name__ == "__main__":
    

    run_alns("instancias/1_hes")
