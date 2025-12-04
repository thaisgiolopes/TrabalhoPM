import sys
import os

# Sobe duas pastas (de evaluation -> alns -> algoritmo)
PASTA_ALGORITMO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PASTA_ALGORITMO)

from solution import Solution
from evaluation.evaluate import evaluate_solution

def swap_workers(sol: Solution, rng):
    data = sol.data
    if data.m < 2: # Cannot swap if fewer than 2 stations
        return False
    a, b = rng.sample(range(data.m), 2)
    sol.worker_by_station[a], sol.worker_by_station[b] = sol.worker_by_station[b], sol.worker_by_station[a]
    evaluate_solution(sol)
    return True

def reassign_worst_station_worker(sol: Solution, rng):
    data = sol.data
    if not sol.Ts or data.m < 2: # Cannot reassign if no station times or fewer than 2 stations
        return False
    # Find the index of the station with the highest cycle time (worst station)
    worst_idx = max(range(len(sol.Ts)), key=lambda s: sol.Ts[s])
    
    # Select a different station randomly to swap workers with
    other_stations = [i for i in range(len(sol.Ts)) if i != worst_idx]
    if not other_stations: # Should not happen if data.m >= 2
        return False
    other = rng.choice(other_stations)

    # Swap the workers between the worst station and the chosen other station
    sol.worker_by_station[worst_idx], sol.worker_by_station[other] = \
        sol.worker_by_station[other], sol.worker_by_station[worst_idx]
    evaluate_solution(sol)
    return True