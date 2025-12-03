# alns_alwabp.py
# Implementação corrigida de ALNS para ALWABP
# Usa apenas bibliotecas padrão: random, math, copy, collections

import random
import math
import copy
from collections import defaultdict, deque

random.seed(0)

# -------------------------
# Função de leitura (você forneceu)
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
                valores.append(int(v))
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

    return n, matriz_tempos, precedencias

# -------------------------
# Conversor de matriz_tempos -> dicionário t[w][i]
# Suponho matriz_tempos como n linhas (tarefas 1..n) e k colunas (trabalhadores 1..k)
# -------------------------
def matriz_para_t(matriz_tempos):
    # matriz_tempos[i-1][w-1] = tempo da tarefa i pelo trabalhador w
    n = len(matriz_tempos)
    if n == 0:
        return {}
    k = len(matriz_tempos[0])
    t = {w: {} for w in range(1, k+1)}
    for i in range(1, n+1):
        for w in range(1, k+1):
            val = matriz_tempos[i-1][w-1]
            t[w][i] = val
    return n, k, t

# -------------------------
# Topological order (Kahn)
# -------------------------
def topological_order(n, precedences):
    adj = defaultdict(list)
    indeg = {i: 0 for i in range(1, n+1)}
    for a, b in precedences:
        # assumimos índices de tarefas válidos
        adj[a].append(b)
        indeg[b] += 1
    q = deque([i for i in indeg if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != n:
        raise ValueError("Ciclo detectado nas precedências (ou número de tarefas inconsistente).")
    return order

# -------------------------
# Cargas e objetivo
# -------------------------
def compute_station_loads(assignment, worker_at_station, t):
    # assignment: dict tarefa -> estação
    # worker_at_station: dict estação -> trabalhador
    m = len(worker_at_station)
    loads = {s: 0.0 for s in range(1, m+1)}
    for i, s in assignment.items():
        w = worker_at_station[s]
        ti = t[w].get(i, float('inf'))
        loads[s] += ti
    return loads

def objective_cycle(assignment, worker_at_station, t):
    loads = compute_station_loads(assignment, worker_at_station, t)
    if not loads:
        return float('inf')
    return max(loads.values())

# -------------------------
# Factibilidade
# -------------------------
def is_feasible_assignment(assignment, worker_at_station, t, precedences):
    # verifica se há tarefas atribuídas a trabalhadores incapazes
    for i, s in assignment.items():
        w = worker_at_station[s]
        if t[w].get(i, float('inf')) == float('inf'):
            return False
    # precedências: station(i) <= station(j) para cada (i,j)
    for a, b in precedences:
        if a not in assignment or b not in assignment:
            # se alguma não atribuída aqui, não podemos afirmar; consideramos não factível no contexto de checagem completa
            return False
        sa = assignment[a]; sb = assignment[b]
        if sa > sb:
            return False
    return True

# -------------------------
# Heurística inicial (greedy respeitando precedências)
# -------------------------
def initial_solution(n, k, t, precedences):
    m = k  # assumimos m = k (uma estação por trabalhador)
    order = topological_order(n, precedences)
    workers = list(range(1, k+1))
    random.shuffle(workers)
    worker_at_station = {s: workers[s-1] for s in range(1, m+1)}
    assignment = {}
    preds_map = defaultdict(list)
    for a,b in precedences:
        preds_map[b].append(a)
    for i in order:
        min_s = 1
        if preds_map[i]:
            # se algum pred não atribuído ainda, assumimos erro — mas order garante preds antes
            min_s = max(assignment[p] for p in preds_map[i])
        placed = False
        for s in range(min_s, m+1):
            w = worker_at_station[s]
            if t[w].get(i, float('inf')) == float('inf'):
                continue
            assignment[i] = s
            placed = True
            break
        if not placed:
            # fallback: procurar qualquer estação factível
            for s in range(1, m+1):
                w = worker_at_station[s]
                if t[w].get(i, float('inf')) != float('inf'):
                    assignment[i] = s
                    placed = True
                    break
        if not placed:
            # se nenhum trabalhador pode executar a tarefa -> atribua à última estação (será infeasible)
            assignment[i] = m
    return worker_at_station, assignment

# -------------------------
# Destroy operators (padronizados)
# assinatura: (assignment, worker_at_station, t, degree) -> (new_assignment, removed_set)
# -------------------------
def destroy_random(assignment, worker_at_station, t, degree):
    tasks = list(assignment.keys())
    rem = set(random.sample(tasks, min(degree, len(tasks))))
    new_assignment = assignment.copy()
    for r in rem:
        del new_assignment[r]
    return new_assignment, rem

def destroy_worst(assignment, worker_at_station, t, degree):
    # remove as tarefas que mais contribuem para a carga (maiores tempos)
    contrib = []
    for i, s in assignment.items():
        w = worker_at_station[s]
        contrib.append((t[w].get(i, float('inf')), i))
    contrib.sort(reverse=True)
    rem = set([i for _, i in contrib[:min(degree, len(contrib))]])
    new_assignment = assignment.copy()
    for r in rem:
        del new_assignment[r]
    return new_assignment, rem

def destroy_by_station(assignment, worker_at_station, t, degree):
    # remove todas as tarefas de algumas estações (escolhe estações com mais carga)
    loads = compute_station_loads(assignment, worker_at_station, t)
    # ordenar estações por carga desc
    stations_sorted = sorted(loads.items(), key=lambda x: x[1], reverse=True)
    removed = set()
    new_assignment = assignment.copy()
    idx = 0
    while len(removed) < degree and idx < len(stations_sorted):
        s = stations_sorted[idx][0]
        # remove todas as tarefas da estação s
        to_remove = [i for i, ss in assignment.items() if ss == s]
        for i in to_remove:
            if i in new_assignment:
                del new_assignment[i]
                removed.add(i)
                if len(removed) >= degree:
                    break
        idx += 1
    return new_assignment, removed

# -------------------------
# Repair operators
# assinatura: (assignment_partial, worker_at_station, t, precedences, unassigned_tasks) -> assignment
# -------------------------
def repair_greedy(assignment_partial, worker_at_station, t, precedences, unassigned_tasks):
    m = len(worker_at_station)
    assignment = assignment_partial.copy()
    preds_map = defaultdict(list)
    for a, b in precedences:
        preds_map[b].append(a)

    unassigned = set(unassigned_tasks)
    # inserir em ordem topológica restrita aos unassigned (e seus pred relativos)
    # construiremos available = tarefas com todos os preds já em assignment
    while unassigned:
        progress = False
        for i in list(unassigned):
            if all(p in assignment for p in preds_map[i]):
                # mínimo s permitido:
                min_s = 1
                if preds_map[i]:
                    min_s = max(assignment[p] for p in preds_map[i])
                best_s = None
                best_obj = float('inf')
                for s in range(min_s, m+1):
                    w = worker_at_station[s]
                    if t[w].get(i, float('inf')) == float('inf'):
                        continue
                    tmp_assign = assignment.copy()
                    tmp_assign[i] = s
                    obj = objective_cycle(tmp_assign, worker_at_station, t)
                    if obj < best_obj:
                        best_obj = obj
                        best_s = s
                if best_s is None:
                    # nenhum trabalhador consegue executar essa tarefa nas estações permitidas agora
                    # adiar (mas se for impossível para todos, atribuir à última estação)
                    feasible_any = False
                    for s in range(1, m+1):
                        w = worker_at_station[s]
                        if t[w].get(i, float('inf')) != float('inf'):
                            feasible_any = True
                            break
                    if feasible_any:
                        # não estava permitido por precedência/estação atual -> deixe para depois
                        continue
                    else:
                        # impossível para qualquer, atribuir a última estação (será infeasible)
                        assignment[i] = m
                        unassigned.remove(i)
                        progress = True
                        continue
                assignment[i] = best_s
                unassigned.remove(i)
                progress = True
        if not progress:
            # para evitar loop infinito (ex: precedências quebradas), force-place remaining arbitrarily em estações factíveis
            for i in list(unassigned):
                placed = False
                for s in range(1, m+1):
                    w = worker_at_station[s]
                    if t[w].get(i, float('inf')) != float('inf'):
                        assignment[i] = s
                        placed = True
                        break
                if not placed:
                    assignment[i] = m
                unassigned.remove(i)
            break
    return assignment

# -------------------------
# Local search: swap trabalhadores entre estações
# -------------------------
def local_search_swap_workers(worker_at_station, assignment, t, precedences, max_tries=50):
    m = len(worker_at_station)
    best_workers = worker_at_station.copy()
    best_assignment = assignment.copy()
    best_obj = objective_cycle(assignment, worker_at_station, t)
    tries = 0
    improved = True
    while tries < max_tries and improved:
        improved = False
        tries += 1
        s1, s2 = random.sample(range(1, m+1), 2)
        new_workers = worker_at_station.copy()
        new_workers[s1], new_workers[s2] = new_workers[s2], new_workers[s1]
        # se nova alocação de trabalhadores tornar impossível respectar precedências -> pular
        if not is_feasible_assignment(assignment, new_workers, t, precedences):
            continue
        new_obj = objective_cycle(assignment, new_workers, t)
        if new_obj < best_obj:
            best_obj = new_obj
            best_workers = new_workers.copy()
            best_assignment = assignment.copy()
            improved = True
    return best_workers, best_assignment, best_obj

# -------------------------
# ALNS main loop
# -------------------------
def alns_solve(n, k, t, precedences,
               iters=2000,
               destroy_ops=None,
               repair_ops=None):
    m = k
    if destroy_ops is None:
        destroy_ops = [destroy_random, destroy_worst, destroy_by_station]
    if repair_ops is None:
        repair_ops = [repair_greedy]

    dest_weights = [1.0] * len(destroy_ops)
    rep_weights = [1.0] * len(repair_ops)
    dest_scores = [0] * len(destroy_ops)
    rep_scores = [0] * len(repair_ops)

    # solução inicial
    worker_at_station, assignment = initial_solution(n, k, t, precedences)
    best_worker = worker_at_station.copy()
    best_assign = assignment.copy()
    best_obj = objective_cycle(assignment, worker_at_station, t)

    current_worker = worker_at_station.copy()
    current_assign = assignment.copy()
    current_obj = best_obj

    T0 = 1.0
    alpha = 0.9995
    T = T0

    for it in range(iters):
        di = random.choices(range(len(destroy_ops)), weights=dest_weights)[0]
        ri = random.choices(range(len(repair_ops)), weights=rep_weights)[0]
        degree = max(1, int(0.2 * n))

        # Destroy: PASSAR worker_at_station e t
        partial_assign, removed = destroy_ops[di](current_assign, current_worker, t, degree)

        # Repair: assinatura conforme implementada
        new_assign = repair_ops[ri](partial_assign, current_worker, t, precedences, removed)

        # Local search: tentar trocar trabalhadores entre estações
        new_workers, new_assign, _ = local_search_swap_workers(current_worker, new_assign, t, precedences, max_tries=10)

        new_obj = objective_cycle(new_assign, new_workers, t)

        # acceptance: simulated annealing
        delta = new_obj - current_obj
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
            current_worker = new_workers.copy()
            current_assign = new_assign.copy()
            current_obj = new_obj
            dest_scores[di] += 1
            rep_scores[ri] += 1
            if current_obj < best_obj and is_feasible_assignment(current_assign, current_worker, t, precedences):
                best_obj = current_obj
                best_worker = current_worker.copy()
                best_assign = current_assign.copy()
                dest_scores[di] += 5
                rep_scores[ri] += 5

        # atualizar pesos a cada 50 iterações
        if (it + 1) % 50 == 0:
            for idx in range(len(dest_weights)):
                dest_weights[idx] = 0.8 * dest_weights[idx] + 0.2 * (1 + dest_scores[idx])
                dest_scores[idx] = 0
            for idx in range(len(rep_weights)):
                rep_weights[idx] = 0.8 * rep_weights[idx] + 0.2 * (1 + rep_scores[idx])
                rep_scores[idx] = 0

        T *= alpha

    return best_worker, best_assign, best_obj

# -------------------------
# Execução de exemplo quando chamado como script
# -------------------------
if __name__ == "__main__":
    # exemplo aleatório gerado localmente (para testes rápidos)
    # se quiser ler de arquivo, use ler_instancia("caminho")
    # Exemplo simples: n=10, k=4 criado aleatoriamente
    def example_instance():
        n = 10
        k = 4
        matriz_tempos = []
        for i in range(n):
            linha = []
            for w in range(k):
                linha.append(random.randint(1, 10))
            matriz_tempos.append(linha)
        # marcar algumas inviabilidades
        matriz_tempos[2][1] = float('inf')  # tarefa 3, trabalhador 2 -> inf
        matriz_tempos[6][2] = float('inf')  # tarefa 7, trabalhador 3 -> inf
        precedencias = [(1,4),(2,4),(4,5),(3,6),(5,7),(6,8),(7,9)]
        return n, matriz_tempos, precedencias

    n, matriz_tempos, precedences = ler_instancia("instancias/1_hes")
    n2, k, t = matriz_para_t(matriz_tempos)
    if n2 != n:
        raise RuntimeError("Inconsistência no número de tarefas após conversão.")
    best_w, best_a, best_c = alns_solve(n, k, t, precedences, iters=2000)
    print("Melhor ciclo (C):", best_c)
    print("Workers por estação:", best_w)
    print("Atribuições (tarefa->estação):", best_a)
