import networkx as nx

class ALWABPData:
    def __init__(self, n, tempos, precedencias):
        self.n = n
        self.tempos = tempos  # [i-1][w-1]
        self.precedencias = precedencias
        self.G = self.criar_grafo_precedencia(n, precedencias)
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

    def criar_grafo_precedencia(n, precedencias): 
        # Grafo de precedência
        G = nx.DiGraph()
        G.add_nodes_from(range(1, n+1))
        G.add_edges_from(precedencias)

        return G

    def twi(self, w, i):
        return self.tempos[i-1][w-1]