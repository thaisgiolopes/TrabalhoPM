import math
import random
import time

from heuristics import (
    initial_solution_greedy, remove_random, remove_worst, remove_shaw,
    insert_greedy, insert_regret, swap_workers_local, reassign_worst_station_worker
)

def accept(old_cost, new_cost, T, rng):
    if new_cost < old_cost:
        return True
    delta = new_cost - old_cost
    prob = math.exp(-delta / max(T, 1e-9))
    return rng.random() < prob

class ALNS:
    def __init__(self, inst, seed=0, segment_length=100, r=0.2,
                 sigma1=10.0, sigma2=5.0, sigma3=1.0, T_start=1.0, cooling=0.995,
                 max_iter=5000, remove_frac=(0.2, 0.4), regret_k=2,
                 time_limit=120, patience=500, refine_callback=None):
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

        self.refine_callback = refine_callback

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

        current = initial_solution_greedy(self.inst, seed=seed_init, max_worker_tries=50)
        if current is None:
            raise RuntimeError("Falha ao construir solução inicial viável.")
        if not current.validate_feasibility():
            raise RuntimeError("Solução inicial inválida pela modelagem.")

        best = current.clone()
        best_cost = best.C
        current_cost = current.C

        visited = set()
        def signature(sol):
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

            # perturbação em trabalhadores com reparo
            if self.rng.random() < 0.3:
                ok_workers = w_op(candidate, self.rng)
                if not ok_workers:
                    # se não conseguiu reparar, descarta perturbação
                    candidate = current.clone()

            # ruin
            q_min = max(4, int(self.remove_frac[0] * self.inst.n))
            q_max = min(100, int(self.remove_frac[1] * self.inst.n))
            q = self.rng.randint(q_min, max(q_min, q_max))
            removed_tasks = r_op(candidate, q, self.rng)

            # recreate por B&B local
            reconstructed = False
            if self.refine_callback is not None and removed_tasks:
                reconstructed = self.refine_callback(candidate, removed_tasks)

            if not reconstructed and removed_tasks:
                ok = i_op(candidate, removed_tasks, self.rng)
                if not ok:
                    ok = insert_greedy(candidate, removed_tasks, self.rng)
                    if not ok:
                        # não reconstruiu; descarta
                        self.T *= self.cooling
                        iter_count += 1
                        segment_counter += 1
                        if segment_counter >= self.segment_length:
                            self.update_weights()
                            segment_counter = 0
                        no_improve += 1
                        continue

            # validação e custo
            candidate.recompute_Ts_full()
            if not candidate.validate_feasibility():
                # solução inválida — descarta
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

            self.T *= self.cooling
            iter_count += 1
            segment_counter += 1
            if segment_counter >= self.segment_length:
                self.update_weights()
                segment_counter = 0

        return best
