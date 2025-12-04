from alwabpData import ALWABPData

def respects_precedence(data: ALWABPData, tasks_by_station):
    # Verifica se para todo (i,j) temos station(i) <= station(j)
    pos_station = {}
    for s_idx, tasks in enumerate(tasks_by_station, start=1):  # estações 1..m
        for i in tasks:
            pos_station[i] = s_idx
    for (i, j) in data.precedencias:
        si = pos_station.get(i)
        sj = pos_station.get(j)
        if si is None or sj is None:
            return False
        if si > sj:
            return False
    return True

def feasible_insertion_station(data: ALWABPData, tasks_by_station, task, station_idx):
    # Checa precedência se task fosse alocada na estação station_idx
    # Basta garantir que para todo predecessor p de task, station(p) <= station_idx
    # e para todo sucessor s de task, station_idx <= station(s)
    pred = list(data.G.predecessors(task))
    succ = list(data.G.successors(task))
    pos_station = {} # Map task -> station_idx
    # Reconstruct pos_station from the current tasks_by_station state
    for s, lst in enumerate(tasks_by_station, start=1):
        for i in lst:
            pos_station[i] = s

    for p in pred:
        sp = pos_station.get(p)
        if sp is None: # Predecessor not yet placed in a station
            # This case means the predecessor must be part of tasks_to_insert
            # or is yet to be considered. We assume it will be placed correctly.
            # For now, it doesn't violate precedence *unless* the predecessor
            # is placed in a station *after* the current station_idx.
            # This function only checks for existing tasks.
            continue
        if sp > station_idx:
            return False # Predecessor is in a later station

    for s in succ:
        ss = pos_station.get(s)
        if ss is None:
            # Similar to predecessors, if a successor is not yet placed,
            # we assume it will be placed correctly later.
            continue
        if station_idx > ss:
            return False # Successor is in an earlier station
    return True