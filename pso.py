from mpi4py import MPI
import numpy as np
import random
import time
import csv
start_time = time.time()  

# Function to read CSV data
def read_csv(file_path):
    data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append([int(cell) for cell in row])
    return data

# Function to generate an initial solution for a particle
def generate_initial_solution(num_jobs, num_machines):
    solution = np.zeros((num_machines, num_jobs), dtype=int)
    for i in range(num_machines):
        solution[i, :] = np.random.permutation(num_jobs)
    return solution

# Function to calculate makespan
def calculate_makespan(individual, processing_times):
    num_machines, num_jobs = individual.shape
    completion_times = np.zeros((num_machines, num_jobs), dtype=int)

    for i in range(num_machines):
        for j in range(num_jobs):
            job_id = individual[i, j]
            if i == 0 and j == 0:
                completion_times[i, j] = processing_times[i][job_id]
            elif i == 0:
                completion_times[i, j] = completion_times[i, j - 1] + processing_times[i][job_id]
            elif j == 0:
                completion_times[i, j] = completion_times[i - 1, j] + processing_times[i][job_id]
            else:
                completion_times[i, j] = max(completion_times[i - 1, j], completion_times[i, j - 1]) + processing_times[i][job_id]

    return completion_times[-1, -1]

# Particle class for PSO
class Particle:
    def __init__(self, num_jobs, num_machines, processing_times):
        self.position = generate_initial_solution(num_jobs, num_machines)
        self.velocity = np.random.uniform(-1, 1, self.position.shape)
        self.best_position = np.copy(self.position)
        self.best_fitness = calculate_makespan(self.position, processing_times)

# Update function for particle velocity
def update_velocity(particle, global_best_position, omega=0.5, phi_p=0.8, phi_g=0.9):
    r_p, r_g = random.random(), random.random()
    cognitive_velocity = phi_p * r_p * (particle.best_position - particle.position)
    social_velocity = phi_g * r_g * (global_best_position - particle.position)
    new_velocity = omega * particle.velocity + cognitive_velocity + social_velocity
    particle.velocity = np.clip(new_velocity, -1, 1)

# Update function for particle position
def update_position(particle, num_jobs, num_machines):
    for i in range(particle.position.shape[0]):
        if random.random() < 0.5:  # Arbitrary threshold for updating position
            swap_index1 = random.randint(0, num_jobs - 1)
            swap_index2 = random.randint(0, num_jobs - 1)
            particle.position[i][swap_index1], particle.position[i][swap_index2] = particle.position[i][swap_index2], particle.position[i][swap_index1]

# Hybrid PSO algorithm with MPI
def hybrid_pso_mpi(num_jobs, num_machines, processing_times, num_particles, num_iterations):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_num_particles = num_particles // size + (1 if rank < num_particles % size else 0)
    local_particles = [Particle(num_jobs, num_machines, processing_times) for _ in range(local_num_particles)]

    if local_particles:
        local_best = min(local_particles, key=lambda x: x.best_fitness)
        global_best_position = local_best.best_position
    else:
        global_best_position = None

    global_best_positions = comm.allgather(global_best_position)
    global_best_fitness = float('inf')
    for pos in global_best_positions:
        if pos is not None:
            fitness = calculate_makespan(pos, processing_times)
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = pos

    for _ in range(num_iterations):
        for particle in local_particles:
            update_velocity(particle, global_best_position)
            update_position(particle, num_jobs, num_machines)
            fitness = calculate_makespan(particle.position, processing_times)
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = np.copy(particle.position)

        local_bests = [(particle.best_position, particle.best_fitness) for particle in local_particles]
        all_bests = comm.allgather(local_bests)

        for bests in all_bests:
            for position, fitness in bests:
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = position

    return global_best_position, global_best_fitness

# Load data from CSV
jobs_processing_times = read_csv('jobsss.csv')  # Replace with your file path
machines_processing_times = read_csv('machinesss.csv')  # Replace with your file path

# Main execution
if __name__ == '__main__':
    num_jobs = 10000  # Number of jobs updated to 10000
    num_machines = 10  # Number of machines updated to 10
    num_particles = 10  # Number of particles (You can adjust this if needed)
    num_iterations = 100  # Number of iterations (You can adjust this if needed)

    best_solution, best_fitness = hybrid_pso_mpi(num_jobs, num_machines, machines_processing_times, num_particles, num_iterations)
    
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Best Solution:", best_solution)
        print("Best Fitness (Makespan):", best_fitness)
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution Time:", execution_time, "seconds")





#mpirun -host 10.100.52.180:2,10.100.53.171:3 -np 5 /usr/bin/python3 pso.py
