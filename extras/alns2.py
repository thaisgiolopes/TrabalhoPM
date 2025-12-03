import sys
import math
import random
import time
from collections import defaultdict
import networkx as nx

# -------------------------
# Leitura da instância
# -------------------------

def ler_instancia(caminho_arquivo):
    with open(caminho_arquivo, "r") as f:
        linhas = [linha.strip() for linha in f if linha.strip()]

    n = int(linhas[0])
    matriz_tempos = []
    idx = 1
    for _ in range(n):
        valores = []
        for v in linhas[idx].split():
            if v.lower() == "inf":
                valores.append(float("inf"))
            else:
                valores.append(float(v))
        matriz_tempos.append(valores)
        idx += 1

    precedencias = []
    while idx < len(linhas):
        i, j = linhas[idx].split()
        i, j = int(i), int(j)
        if i == -1 and j == -1:
            break
        precedencias.append((i, j))
        idx += 1

    # Grafo de precedência para validação e topológica (apenas no início)
    G = nx.DiGraph()
    G.add_nodes_from(range(1, n+1))
    G.add_edges_from(precedencias)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Instância inválida: o grafo de precedência contém ciclos.")

    return n, matriz_tempos, precedencias, G

# -------------------------
# Dados do problema com pré-cálculos
# -------------------------

class ALWABPData:
    def __init__(self, n, tempos, precedencias, G):
        self.n = n
        self.tempos = tempos  # [i-1][w-1]
        self.precedencias = precedencias
        self.G = G
        self.k = len(tempos[0])         # trabalhadores
        self.m = self.k                 # estações (versão clássica)
        self.tarefas = list(range(1, n+1))
        self.trabalhadores = list(range(1, self.k+1))
        self.estacoes = list(range(1, self.m+1))

        # Incapacidade por trabalhador
        self.incapacidade = {w: set() for w in self.trabalhadores}
        for i in self.tarefas:
            for w in self.trabalhadores:
                if self.tempos[i-1][w-1] == float("inf"):
                    self.incapacidade[w].add(i)

        # Pré-calcular predecessores e sucessores
        self.pred = {i: set(G.predecessors(i)) for i in self.tarefas}
        self.succ = {i: set(G.successors(i)) for i in self.tarefas}

        # Ordem topológica única e índice
        self.topo_order = list(nx.topological_sort(G))
        self.topo_index = {i: idx for idx, i in enumerate(self.topo_order)}

        # Tempos médios por tarefa (ignorando inf) para Shaw rápido
        self.mean_time = {}
        for i in self.tarefas:
            vals = [self.tempos[i-1][w-1] for w in self.trabalhadores if self.tempos[i-1][w-1] != float("inf")]
            self.mean_time[i] = (sum(vals) / len(vals)) if vals else float("inf")

    def twi(self, w, i):
        return self.tempos[i-1][w-1]

# -------------------------
# Representação da solução com estruturas incrementais
# -------------------------

class Solution:
    def __init__(self, data: ALWABPData):
        self.data = data
        self.tasks_by_station = [[] for _ in data.estacoes]  # index 0..m-1 para estação 1..m
        self.worker_by_station = [None for _ in data.estacoes]
        self.Ts = [0.0 for _ in data.estacoes]
        self.C = float("inf")
        # posição por tarefa: estação 1..m, ou None
        self.position = {i: None for i in data.tarefas}

    def clone(self):
        s2 = Solution(self.data)
        s2.tasks_by_station = [list(lst) for lst in self.tasks_by_station]
        s2.worker_by_station = list(self.worker_by_station)
        s2.Ts = list(self.Ts)
        s2.C = self.C
        s2.position = dict(self.position)
        return s2

    # Operações incrementais

    def insert_task(self, i, station_idx):
        """Insere tarefa i na estação station_idx (1..m). Atualiza Ts e position."""
        s = station_idx - 1
        w = self.worker_by_station[s]
        t = self.data.twi(w, i)
        if t == float("inf"):
            return False
        self.tasks_by_station[s].append(i)
        self.Ts[s] += t
        self.position[i] = station_idx
        self.C = max(self.C, self.Ts[s])
        return True

    def remove_task(self, i):
        """Remove tarefa i de sua estação atual. Atualiza Ts e position."""
        st = self.position[i]
        if st is None:
            return False
        s = st - 1
        w = self.worker_by_station[s]
        t = self.data.twi(w, i)
        # Remover da lista (O(len)), aceitável
        self.tasks_by_station[s].remove(i)
        self.Ts[s] -= t if t != float("inf") else 0.0
        self.position[i] = None
        # Atualiza C de forma segura
        self.C = max(self.Ts) if self.Ts else float("inf")
        return True

    def set_workers(self, workers_list):
        """Define trabalhadores por estação e recalcula Ts incrementalmente."""
        self.worker_by_station = list(workers_list)
        # Recalcula Ts eficiente: soma por estação, usando tempos dos novos workers
        self.Ts = [0.0 for _ in self.data.estacoes]
        for s_idx in range(self.data.m):
            w = self.worker_by_station[s_idx]
            total = 0.0
            bad = False
            for i in self.tasks_by_station[s_idx]:
                t = self.data.twi(w, i)
                if t == float("inf"):
                    bad = True
                    break
                total += t
            self.Ts[s_idx] = float("inf") if bad else total
        self.C = max(self.Ts) if self.Ts else float("inf")

# -------------------------
# Viabilidade de precedência rápida
# -------------------------

def feasible_insertion_station(sol: Solution, task, station_idx):
    """Checa precedência para inserir task na station_idx usando pred/succ e sol.position."""
    data = sol.data
    # predecessores devem estar em estação <= station_idx
    for p in data.pred[task]:
        sp = sol.position[p]
        if sp is not None and sp > station_idx:
            return False
    # sucessores devem estar em estação >= station_idx
    for s in data.succ[task]:
        ss = sol.position[s]
        if ss is not None and station_idx > ss:
            return False
    return True

# -------------------------
# Construção de solução inicial (otimizada)
# -------------------------

def initial_solution_greedy(data: ALWABPData, seed=0):
    rng = random.Random(seed)
    sol = Solution(data)

    # Atribui trabalhadores às estações (embaralhar para diversificação)
    workers = list(data.trabalhadores)
    rng.shuffle(workers)
    sol.set_workers(workers)

    # Inserção na ordem topológica
    for i in data.topo_order:
        best_s, best_cost = None, float("inf")
        for station_idx in range(1, data.m+1):
            if not feasible_insertion_station(sol, i, station_idx):
                continue
            s = station_idx - 1
            w = sol.worker_by_station[s]
            t = data.twi(w, i)
            if t == float("inf"):
                continue
            new_load = sol.Ts[s] + t
            proxy_C = max(new_load, sol.C if sol.C != float("inf") else 0.0)
            if proxy_C < best_cost:
                best_cost = proxy_C
                best_s = station_idx
        if best_s is None:
            return None
        sol.insert_task(i, best_s)

    return sol

# -------------------------
# Operadores de remoção (incrementais)
# -------------------------

def remove_random(sol: Solution, q, rng):
    removed = []
    all_tasks = [i for lst in sol.tasks_by_station for i in lst]
    rng.shuffle(all_tasks)
    for i in all_tasks[:q]:
        sol.remove_task(i)
        removed.append(i)
    return removed

def remove_worst(sol: Solution, q, rng):
    data = sol.data
    contrib = []
    for s_idx in range(sol.data.m):
        w = sol.worker_by_station[s_idx]
        for i in sol.tasks_by_station[s_idx]:
            contrib.append((data.twi(w, i), i))
    contrib.sort(reverse=True)
    removed = []
    for _, i in contrib[:q]:
        sol.remove_task(i)
        removed.append(i)
    return removed

def remove_shaw(sol: Solution, q, rng):
    data = sol.data
    all_tasks = [i for lst in sol.tasks_by_station for i in lst]
    if not all_tasks:
        return []
    seed_task = rng.choice(all_tasks)
    # Relatedness leve: proximidade topológica + diferença de tempo médio normalizada
    def relatedness(a, b):
        da = data.topo_index[a]
        db = data.topo_index[b]
        dtopo = abs(da - db)
        ma = data.mean_time[a]
        mb = data.mean_time[b]
        if ma == float("inf") or mb == float("inf"):
            dtime = 1.0
        else:
            denom = max(ma + mb, 1e-9)
            dtime = abs(ma - mb) / denom
        return dtopo + dtime
    ranked = sorted(all_tasks, key=lambda t: relatedness(seed_task, t))
    removed = []
    for i in ranked[:q]:
        sol.remove_task(i)
        removed.append(i)
    return removed

# -------------------------
# Operadores de inserção (incrementais)
# -------------------------

def insert_greedy(sol: Solution, tasks_to_insert, rng, noise=0.0):
    for i in tasks_to_insert:
        best_s, best_cost = None, float("inf")
        for station_idx in range(1, sol.data.m+1):
            if not feasible_insertion_station(sol, i, station_idx):
                continue
            s = station_idx - 1
            w = sol.worker_by_station[s]
            t = sol.data.twi(w, i)
            if t == float("inf"):
                continue
            new_load = sol.Ts[s] + t
            proxy_C = max(new_load, sol.C)
            proxy_C += noise * rng.random()
            if proxy_C < best_cost:
                best_cost = proxy_C
                best_s = station_idx
        if best_s is None:
            return False
        sol.insert_task(i, best_s)
    return True

def insert_regret(sol: Solution, tasks_to_insert, rng, k=2, noise=0.0):
    pending = set(tasks_to_insert)
    if not pending:
        return True

    while pending:
        best_opts = {}
        for i in list(pending):
            opts = []
            for station_idx in range(1, sol.data.m+1):
                if not feasible_insertion_station(sol, i, station_idx):
                    continue
                s = station_idx - 1
                w = sol.worker_by_station[s]
                t = sol.data.twi(w, i)
                if t == float("inf"):
                    continue
                new_load = sol.Ts[s] + t
                proxy_C = max(new_load, sol.C) + noise * rng.random()
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
        
        # This check is crucial and should prevent the error if chosen_station is None.
        if chosen_task is None or chosen_station is None:
            return False

        sol.insert_task(chosen_task, chosen_station)
        pending.remove(chosen_task)
    return True

# -------------------------
# Operadores sobre trabalhadores (incrementais)
# -------------------------

def swap_workers(sol: Solution, rng):
    a, b = rng.sample(range(sol.data.m), 2)
    sol.worker_by_station[a], sol.worker_by_station[b] = sol.worker_by_station[b], sol.worker_by_station[a]
    # Recalcular cargas das duas estações apenas
    for s in (a, b):
        w = sol.worker_by_station[s]
        total = 0.0
        bad = False
        for i in sol.tasks_by_station[s]:
            t = sol.data.twi(w, i)
            if t == float("inf"):
                bad = True
                break
            total += t
        sol.Ts[s] = float("inf") if bad else total
    sol.C = max(sol.Ts)
    return True

def reassign_worst_station_worker(sol: Solution, rng):
    worst_idx = max(range(len(sol.Ts)), key=lambda s: sol.Ts[s])
    other = rng.choice([i for i in range(len(sol.Ts)) if i != worst_idx])
    return swap_workers(sol, rng)  # reaproveita a lógica de recalcular localmente

# -------------------------
# Critério de aceitação (SA)
# -------------------------

def accept(old_cost, new_cost, T, rng):
    if new_cost < old_cost:
        return True
    delta = new_cost - old_cost
    prob = math.exp(-delta / max(T, 1e-9))
    return rng.random() < prob

# -------------------------
# ALNS framework (otimizado)
# -------------------------

class ALNS:
    def __init__(self, data: ALWABPData, seed=0, segment_length=100, r=0.2,
                 sigma1=10.0, sigma2=5.0, sigma3=1.0, T_start=None, cooling=0.995,
                 max_iter=5000, remove_frac=(0.2, 0.4), noise=0.0, regret_k=2, time_limit=None):
        self.data = data
        self.rng = random.Random(seed)
        self.segment_length = segment_length
        self.r = r
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.cooling = cooling
        self.max_iter = max_iter
        self.remove_frac = remove_frac
        self.noise = noise
        self.regret_k = regret_k
        self.time_limit = time_limit
        self.T = T_start

        self.remove_ops = [
            ("random_remove", remove_random),
            ("worst_remove", remove_worst),
            ("shaw_remove", remove_shaw),
        ]
        self.insert_ops = [
            ("greedy_insert", insert_greedy),
            ("regret_insert", lambda sol, tasks, rng, noise: insert_regret(sol, tasks, rng, k=self.regret_k, noise=noise)),
        ]
        self.worker_ops = [
            ("swap_workers", swap_workers),
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

        current = initial_solution_greedy(self.data, seed=seed_init)
        if current is None:
            for s in range(1, 20):
                current = initial_solution_greedy(self.data, seed=seed_init + s)
                if current is not None:
                    break
            if current is None:
                raise RuntimeError("Falha ao construir solução inicial viável.")

        best = current.clone()
        best_cost = best.C
        current_cost = current.C

        # Temperatura inicial baseada em pequenas perturbações
        if self.T is None:
            samples = []
            for _ in range(15):
                tmp = current.clone()
                removed = remove_random(tmp, max(1, int(0.1 * self.data.n)), self.rng)
                insert_greedy(tmp, removed, self.rng, noise=self.noise)
                samples.append(tmp.C)
            mean = sum(samples) / len(samples)
            stdev = (sum((c - mean) ** 2 for c in samples) / max(len(samples) - 1, 1)) ** 0.5
            self.T = max(stdev, 1.0)

        visited = set()
        def signature(sol):
            return tuple((sol.worker_by_station[s], tuple(sorted(sol.tasks_by_station[s]))) for s in range(sol.data.m))
        visited.add(signature(current))

        iter_count = 0
        segment_counter = 0

        while iter_count < self.max_iter:
            if self.time_limit and (time.time() - start_time) >= self.time_limit:
                break

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

            # Pequeno passo em trabalhadores com prob. moderada
            if self.rng.random() < 0.3:
                w_op(candidate, self.rng)

            # Remoção/Inserção grandes
            q_min = max(4, int(self.remove_frac[0] * self.data.n))
            q_max = min(100, int(self.remove_frac[1] * self.data.n))
            q = self.rng.randint(q_min, max(q_min, q_max))

            removed_tasks = r_op(candidate, q, self.rng)

            ok = i_op(candidate, removed_tasks, self.rng, noise=self.noise)
            if not ok:
                ok = insert_greedy(candidate, removed_tasks, self.rng, noise=self.noise)
                if not ok:
                    self.T *= self.cooling
                    iter_count += 1
                    segment_counter += 1
                    if segment_counter >= self.segment_length:
                        self.update_weights()
                        segment_counter = 0
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
                self.remove_scores[ridx] += self.sigma1
                self.insert_scores[iidx] += self.sigma1
                self.worker_scores[widx] += self.sigma1
            elif is_better_than_current and is_new_solution:
                self.remove_scores[ridx] += self.sigma2
                self.insert_scores[iidx] += self.sigma2
                self.worker_scores[widx] += self.sigma2
            elif accepted and is_new_solution:
                self.remove_scores[ridx] += self.sigma3
                self.insert_scores[iidx] += self.sigma3
                self.worker_scores[widx] += self.sigma3

            if is_new_solution:
                visited.add(sig)

            self.T *= self.cooling
            iter_count += 1
            segment_counter += 1

            if segment_counter >= self.segment_length:
                self.update_weights()
                segment_counter = 0

        return best

# -------------------------
# CLI simples para executar ALNS
# -------------------------

def run_alns(instancia_path, seed=0, time_limit=None, max_iter=5000, segment_length=100,
             cooling=0.995, noise=0.0, regret_k=2, remove_frac_low=0.2, remove_frac_high=0.4):
    n, tempos, precedencias, G = ler_instancia(instancia_path)
    data = ALWABPData(n, tempos, precedencias, G)
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

    print(f"Tempo de ciclo (ALNS) C: {best.C:.6f}")
    for s_idx in range(data.m):
        w = best.worker_by_station[s_idx]
        tarefas = sorted(best.tasks_by_station[s_idx])
        Ts = best.Ts[s_idx]
        print(f"Estação {s_idx+1}: trabalhador {w}, T_s = {Ts:.6f}, tarefas: {tarefas}")

    return best

run_alns("instancias/11_wee")