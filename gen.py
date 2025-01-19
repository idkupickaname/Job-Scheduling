import numpy as np
import random
import time
import csv
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
start_time = time.time()  

def read_csv(file_path):
    data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append([int(cell) for cell in row])
    return data

def generate_initial_population(num_jobs, num_machines, population_size):
    population = []
    for _ in range(population_size):
        solution = np.zeros((num_machines, num_jobs), dtype=int)
        for i in range(num_machines):
            solution[i, :] = np.random.permutation(num_jobs)
        population.append(solution)
    return population

def calculate_makespan(individual, processing_times):
    num_machines, num_jobs = individual.shape
    completion_times = np.zeros((num_machines, num_jobs), dtype=int)

    for i in range(num_machines):
        for j in range(num_jobs):
            if i == 0 and j == 0:
                completion_times[i, j] = processing_times[i][individual[i, j]]
            elif i == 0:
                completion_times[i, j] = completion_times[i, j - 1] + processing_times[i][individual[i, j]]
            elif j == 0:
                completion_times[i, j] = completion_times[i - 1, j] + processing_times[i][individual[i, j]]
            else:
                completion_times[i, j] = max(completion_times[i - 1, j], completion_times[i, j - 1]) + processing_times[i][individual[i, j]]

    return completion_times[-1, -1]

def genetic_algorithm_mpi(num_jobs, num_machines, processing_times, population_size, num_generations, crossover_rate, mutation_rate):
    local_population_size = population_size // size
    local_population = generate_initial_population(num_jobs, num_machines, local_population_size)

    makespan_history = []

    for generation in range(num_generations):
        local_fitness_scores = [1 / calculate_makespan(individual, processing_times) for individual in local_population]

        # Gather fitness scores and combine them at the root process
        all_fitness_scores = comm.gather(local_fitness_scores, root=0)

        if rank == 0:
            # Flatten the list of fitness scores
            all_fitness_scores_flat = [score for sublist in all_fitness_scores for score in sublist]
            
            # Select parents proportionally to the entire population, not just the local_population
            parents = random.choices(local_population * size, weights=all_fitness_scores_flat, k=local_population_size * size)
        else:
            parents = None

        # Broadcast parents from root to all processes
        parents = comm.bcast(parents, root=0)
        
        # Each process takes a subset of parents for offspring generation
        local_parents = parents[rank * local_population_size: (rank + 1) * local_population_size]

        # Generate offspring locally
        offspring = []
        for parent1, parent2 in zip(local_parents[::2], local_parents[1::2]):
            if random.random() < crossover_rate:
                crossover_point = random.randint(1, num_jobs - 1)
                child1 = np.hstack((parent1[:, :crossover_point], parent2[:, crossover_point:]))
                child2 = np.hstack((parent2[:, :crossover_point], parent1[:, crossover_point:]))
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                mutation_point = random.randint(0, num_jobs - 1)
                machine_idx = random.randint(0, num_machines - 1)
                child1[machine_idx, mutation_point] = random.sample(range(num_jobs), 1)[0]
                child2[machine_idx, mutation_point] = random.sample(range(num_jobs), 1)[0]

            offspring.extend([child1, child2])

        local_population = offspring

        # Find the best individual in the local population for this generation
        best_individual_local = min(local_population, key=lambda x: calculate_makespan(x, processing_times))
        best_makespan_local = calculate_makespan(best_individual_local, processing_times)

        # Gather best makespans and find the overall best
        all_best_makespans = comm.gather(best_makespan_local, root=0)
        if rank == 0:
            overall_best_makespan = min(all_best_makespans)
            makespan_history.append(overall_best_makespan)

    # Finalizing the algorithm
    if rank == 0:
        # The root process returns the final results
        return best_individual_local, overall_best_makespan, makespan_history
    else:
        # Other processes return None
        return None, None, None

# Generate processing times for jobs and machines
num_jobs = 10000
num_machines = 10
jobs_processing_times = np.random.randint(1, 100, size=(num_jobs, num_machines))
machines_processing_times = np.random.randint(1, 100, size=(num_machines, num_jobs))

# Perform scheduling using Genetic Algorithm
population_size = 50
num_generations = 100
crossover_rate = 0.8
mutation_rate = 0.1

best_solution, best_makespan, makespan_history = genetic_algorithm_mpi(
num_jobs, num_machines, machines_processing_times,
population_size, num_generations, crossover_rate, mutation_rate
)


if MPI.COMM_WORLD.Get_rank() == 0:
    print("Best Solution:", best_solution)
    print("Best Makespan:", best_makespan)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution Time:", execution_time, "seconds")


# # Print best solution and makespan
# print("Best Solution:")
# print(best_solution)
# print("Best Makespan:", best_makespan)
# end_time = time.time()
# execution_time = end_time - start_time
# print("Execution Time:", execution_time, "seconds")
