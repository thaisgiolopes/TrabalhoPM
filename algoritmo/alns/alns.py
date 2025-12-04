import math
import random
import time
import sys
import os

# Sobe duas pastas (de evaluation -> alns -> algoritmo)
PASTA_ALGORITMO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PASTA_ALGORITMO)

from alwabpData import ALWABPData

from alns.evaluation.evaluate import evaluate_solution
from alns.heuristics.construction import initial_solution_greedy
from alns.operators.insertions import insert_greedy, insert_regret
from alns.operators.removals import remove_random, remove_shaw, remove_worst
from alns.operators.workers import reassign_worst_station_worker, swap_workers

def accept(old_cost, new_cost, T, rng):
        if new_cost < old_cost:
            return True
        delta = new_cost - old_cost
        prob = math.exp(-delta / max(T, 1e-9))
        return rng.random() < prob

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

        # Registro de heurísticas
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
        # Pesos e pontuações
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
        # Atualização adaptativa por segmento
        for arr_w, arr_sc, arr_use in [
            (self.remove_weights, self.remove_scores, self.remove_uses),
            (self.insert_weights, self.insert_scores, self.insert_uses),
            (self.worker_weights, self.worker_scores, self.worker_uses),
        ]:
            for i in range(len(arr_w)):
                if arr_use[i] > 0:
                    arr_w[i] = (1 - self.r) * arr_w[i] + self.r * (arr_sc[i] / arr_use[i])
                # reset pontuação e usos para próximo segmento
                arr_sc[i] = 0.0
                arr_use[i] = 0

    def run(self, seed_init=0):
        start_time = time.time()

        # Construir solução inicial
        current = initial_solution_greedy(self.data, seed=seed_init)
        if current is None:
            # Tentar algumas vezes com diferentes seeds
            for s in range(1, 20):
                current = initial_solution_greedy(self.data, seed=seed_init + s)
                if current is not None:
                    break
            if current is None:
                raise RuntimeError("Falha ao construir solução inicial viável.")

        solucao_inicial = current.clone()
        best = current.clone()
        best_cost = best.C
        current_cost = current.C
        best_time = 0.0  # tempo até encontrar melhor solução

        # Temperatura inicial
        if self.T is None:
            samples = []
            for _ in range(20):
                tmp = current.clone()
                if len(self.data.tarefas) >= 2:
                    removed = remove_random(tmp, max(1, int(0.1 * self.data.n)), self.rng)
                    if removed:
                        insert_greedy(tmp, removed, self.rng, noise=self.noise)
                evaluate_solution(tmp)
                samples.append(tmp.C)
            if len(samples) > 1 and not all(c == samples[0] for c in samples):
                stdev = (sum((c - sum(samples)/len(samples))**2 for c in samples) / (len(samples)-1))**0.5
                self.T = max(stdev, 1.0)
            else:
                self.T = 10.0

        visited = set()
        def signature(sol):
            sig = []
            for s_idx in range(self.data.m):
                sig.append((sol.worker_by_station[s_idx], tuple(sorted(sol.tasks_by_station[s_idx]))))
            return tuple(sig)

        visited.add(signature(current))

        iter_count = 0
        segment_counter = 0

        while iter_count < self.max_iter:
            if self.time_limit and (time.time() - start_time) >= self.time_limit:
                break

            # Seleções por roleta
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

            if self.data.m >= 2 and self.rng.random() < 0.3:
                w_op(candidate, self.rng)

            q_min = max(1, int(self.remove_frac[0] * self.data.n))
            q_max = min(self.data.n, int(self.remove_frac[1] * self.data.n))
            q = self.rng.randint(q_min, max(q_min, q_max))

            removed_tasks = r_op(candidate, q, self.rng)

            ok = i_op(candidate, removed_tasks, self.rng, noise=self.noise)
            if not ok:
                ok = insert_greedy(candidate, removed_tasks, self.rng, noise=self.noise)
                if not ok:
                    iter_count += 1
                    segment_counter += 1
                    self.T *= self.cooling
                    if segment_counter >= self.segment_length:
                        self.update_weights()
                        segment_counter = 0
                    continue

            evaluate_solution(candidate)
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
                best_time = time.time() - start_time  # registra tempo até achar melhor
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

        return solucao_inicial, best, best_time
