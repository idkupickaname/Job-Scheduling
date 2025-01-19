import csv
import random

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def read_csv(file_path):
    data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append([int(cell) for cell in row])
    return data

def generate_initial_solution(num_jobs, num_machines):
    initial_solution = [[j for j in range(num_jobs)] for _ in range(num_machines)]
    for machine_schedule in initial_solution:
        random.shuffle(machine_schedule)
    return initial_solution

def calculate_makespan(solution, processing_times):
    num_jobs = len(solution[0])
    num_machines = len(solution)
    completion_times = [[0] * num_jobs for _ in range(num_machines)]

    for i in range(num_machines):
        for j in range(num_jobs):
            if i == 0 and j == 0:
                completion_times[i][j] = processing_times[solution[i][j]][i]
            elif i == 0:
                completion_times[i][j] = completion_times[i][j - 1] + processing_times[solution[i][j]][i]
            elif j == 0:
                completion_times[i][j] = completion_times[i - 1][j] + processing_times[solution[i][j]][i]
            else:
                completion_times[i][j] = max(completion_times[i - 1][j], completion_times[i][j - 1]) + processing_times[solution[i][j]][i]

    return completion_times[-1][-1]

def perturb_solution(solution):
    num_machines = len(solution)
    for i in range(num_machines):
        j1, j2 = random.sample(range(len(solution[i])), 2)
        solution[i][j1], solution[i][j2] = solution[i][j2], solution[i][j1]
    return solution

def iterated_greedy_mpi(num_jobs, num_machines, processing_times, max_iterations):
    local_best_solution = generate_initial_solution(num_jobs, num_machines)
    local_best_makespan = calculate_makespan(local_best_solution, processing_times)
    
    for iteration in range(max_iterations):
        perturbed_solution = perturb_solution(local_best_solution)
        perturbed_makespan = calculate_makespan(perturbed_solution, processing_times)

        if perturbed_makespan < local_best_makespan:
            local_best_solution = perturbed_solution
            local_best_makespan = perturbed_makespan

    # Gather all solutions and makespans at the root process
    all_best_solutions = comm.gather(local_best_solution, root=0)
    all_best_makespans = comm.gather(local_best_makespan, root=0)

    # Root process finds the overall best solution and makespan
    if rank == 0:
        overall_best_index = all_best_makespans.index(min(all_best_makespans))
        overall_best_solution = all_best_solutions[overall_best_index]
        overall_best_makespan = all_best_makespans[overall_best_index]
        return overall_best_solution, overall_best_makespan
    else:
        return None, None

if rank == 0:
    jobs_processing_times = read_csv('jobs.csv')
    machines_processing_times = read_csv('machines.csv')

    expected_num_jobs = 1000
    expected_num_machines = 3

    if len(jobs_processing_times) != expected_num_jobs:
        print(f"Error: Expected {expected_num_jobs} jobs, but got {len(jobs_processing_times)} jobs.")
    elif len(machines_processing_times) != expected_num_machines:
        print(f"Error: Expected {expected_num_machines} machines, but got {len(machines_processing_times)} machines.")
    else:
        print("Data loaded successfully.")
        max_iterations = 5

        best_solution, best_makespan = iterated_greedy_mpi(
        expected_num_jobs, expected_num_machines, jobs_processing_times, max_iterations
    )

        print("\nFinal Best Solution (Output):")

        print("\nBest Makespan:", best_makespan)

