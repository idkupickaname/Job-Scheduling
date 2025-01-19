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
    local_population_size = population_size // size
    local_population = []
    for _ in range(local_population_size):
        solution = np.zeros((num_machines, num_jobs), dtype=int)
        for i in range(num_machines):
            solution[i, :] = np.random.permutation(num_jobs)
        local_population.append(solution)
    return local_population

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






def update_velocity_pso(velocity, position, personal_best, global_best, omega, phi_p, phi_g):
    r_p, r_g = random.random(), random.random()
    cognitive_velocity = phi_p * r_p * (personal_best - position)
    social_velocity = phi_g * r_g * (global_best - position)
    new_velocity = omega * velocity + cognitive_velocity + social_velocity
    return np.clip(new_velocity, -1, 1)

def update_position_pso(position, velocity, num_jobs):
    for i in range(position.shape[0]):
        if random.random() < 0.5:
            swap_index1 = random.randint(0, num_jobs - 1)
            swap_index2 = random.randint(0, num_jobs - 1)
            position[i][swap_index1], position[i][swap_index2] = position[i][swap_index2], position[i][swap_index1]


def hybrid_ga_pso_mpi(num_jobs, num_machines, machines_processing_times, population_size, num_iterations, crossover_rate, mutation_rate, omega, phi_p, phi_g):
    local_population = generate_initial_population(num_jobs, num_machines, population_size)
    local_velocities = [np.random.uniform(-1, 1, (num_machines, num_jobs)) for _ in range(len(local_population))]
    
    # Initial local best solution
    local_best_idx, local_best_solution = min(enumerate(local_population), key=lambda x: calculate_makespan(x[1], machines_processing_times))
    local_best_makespan = calculate_makespan(local_best_solution, machines_processing_times)
    local_best_tuple = (local_best_makespan, rank)

    # Initial global best solution
    global_best_tuple = comm.allreduce(local_best_tuple, op=MPI.MINLOC)
    global_best_solution = None
    if rank == global_best_tuple[1]:
        global_best_solution = local_population[local_best_idx]
    global_best_solution = comm.bcast(global_best_solution, root=global_best_tuple[1])
    
    for iteration in range(num_iterations):
        # PSO Step
        for i, individual in enumerate(local_population):
            local_velocities[i] = update_velocity_pso(local_velocities[i], individual, local_best_solution, global_best_solution, omega, phi_p, phi_g)
            update_position_pso(individual, local_velocities[i], num_jobs)

        # GA Step
        for i, individual in enumerate(local_population):
            if random.random() < crossover_rate:
                partner_index = random.randint(0, len(local_population) - 1)
                child1, child2 = crossover_and_mutation(individual, local_population[partner_index], mutation_rate, num_jobs, num_machines)
                local_population[i] = child1
                local_population[partner_index] = child2

        # Local Evaluation and Update Bests
        local_best_idx, local_best_solution = min(enumerate(local_population), key=lambda x: calculate_makespan(x[1], machines_processing_times))
        local_best_makespan = calculate_makespan(local_best_solution, machines_processing_times)
        local_best_tuple = (local_best_makespan, rank)

        # Update Global Bests
        global_best_tuple = comm.allreduce(local_best_tuple, op=MPI.MINLOC)
        global_best_solution = None
        if rank == global_best_tuple[1]:
            global_best_solution = local_population[local_best_idx]
        global_best_solution = comm.bcast(global_best_solution, root=global_best_tuple[1])

    if rank == 0:
        return global_best_solution, calculate_makespan(global_best_solution, machines_processing_times)
    else:
        return None, None


def crossover_and_mutation(parent1, parent2, mutation_rate, num_jobs, num_machines):
    crossover_point = random.randint(1, num_jobs - 1)
    child1 = np.hstack((parent1[:, :crossover_point], parent2[:, crossover_point:]))
    child2 = np.hstack((parent2[:, :crossover_point], parent1[:, crossover_point:]))

    if random.random() < mutation_rate:
        mutation_point = random.randint(0, num_jobs - 1)
        machine_idx = random.randint(0, num_machines - 1)
        child1[machine_idx, mutation_point] = random.sample(range(num_jobs), 1)[0]
        child2[machine_idx, mutation_point] = random.sample(range(num_jobs), 1)[0]

    return child1, child2

# Generate processing times for jobs and machines
if __name__ == "__main__":
    num_jobs = 9999
    num_machines = 10
    machines_processing_times = np.random.randint(1, 100, size=(num_machines, num_jobs))

    population_size = 50
    num_iterations = 100
    crossover_rate = 0.8
    mutation_rate = 0.1
    omega = 0.5
    phi_p = 0.8
    phi_g = 0.9

    best_solution, best_makespan = hybrid_ga_pso_mpi(
        num_jobs, num_machines, machines_processing_times,
        population_size, num_iterations, crossover_rate, mutation_rate,
        omega, phi_p, phi_g
    )

    if rank == 0:
        print("Best Solution:", best_solution)
        print("Best Makespan:", best_makespan)
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution Time:", execution_time, "seconds")

