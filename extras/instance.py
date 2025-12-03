from collections import deque
from ler_instancia import ler_instancia

class Instance:
    def __init__(self, path):
        self.n, self.tempos, self.precedencias = ler_instancia(path)
        self.tarefas = list(range(1, self.n + 1))
        self.k = len(self.tempos[0])
        self.trabalhadores = list(range(1, self.k + 1))
        self.m = self.k
        self.estacoes = list(range(1, self.m + 1))

        # Incapacidade por trabalhador
        self.incapacidade = {w: set() for w in self.trabalhadores}
        for i in self.tarefas:
            for w in self.trabalhadores:
                if self.tempos[i-1][w-1] == float("inf"):
                    self.incapacidade[w].add(i)

        # Predecessores e sucessores
        self.pred = {i: set() for i in self.tarefas}
        self.succ = {i: set() for i in self.tarefas}
        for (i, j) in self.precedencias:
            self.succ[i].add(j)
            self.pred[j].add(i)

        # Ordem topológica (Kahn)
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
            raise ValueError("Instância inválida: ciclo em precedências.")
        topo_index = {i: idx for idx, i in enumerate(order)}
        return order, topo_index

    def twi(self, w, i):
        return self.tempos[i-1][w-1]
