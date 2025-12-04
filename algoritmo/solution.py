from alwabpData import ALWABPData

class Solution:
    def __init__(self, data: ALWABPData):
        self.data = data
        self.tasks_by_station = [[] for _ in data.estacoes]  # index 0..m-1 para estação 1..m
        self.worker_by_station = [None for _ in data.estacoes]  # trabalhador alocado a cada estação
        self.Ts = [0.0 for _ in data.estacoes]
        self.C = float("inf")

    def clone(self):
        s2 = Solution(self.data)
        s2.tasks_by_station = [list(lst) for lst in self.tasks_by_station]
        s2.worker_by_station = list(self.worker_by_station)
        s2.Ts = list(self.Ts)
        s2.C = self.C
        return s2