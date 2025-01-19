# Job Scheduling using Meta Heuristic Approach

## Overview
This project focuses on optimizing job scheduling using meta-heuristic approaches, specifically the Iterated Greedy Algorithm, Genetic Algorithm and the Particle Swarm Optimization algorithm. The application is developed to operate in a distributed computing environment, utilizing MPI4PY for parallel processing.

## Features
- **Meta-Heuristic Algorithms**: Utilizes Iterated Greedy and Genetic Algorithm for effective job scheduling.
- **Distributed Computing**: Leverages MPI4PY for distributed task execution, enhancing performance.
- **Performance Testing**: Includes comparisons of job scheduling efficiency in single-computer and distributed setups.

## Technologies and Tools
- **Algorithms**: Iterated Greedy Algorithm, Genetic Algorithm
- **Distributed Computing**: MPI4PY
- **Programming Language**: Python

## Datasets
- **Dataset 1**: 10,000 jobs across 10 machines
- **Dataset 2**: 1,000 jobs across 3 machines

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```sh
   git clone https://github.com/drohan0717/Job-Scheduling.git
   
2. **Setup your master and slave nodes where master represents the node which initiates execution, follow this playlist for in detail understanding for setting up your distributed system.**
   [https://www.youtube.com/watch?v=-t4k6IwmtFI&t=75s]

3. **Install MPI4PY to provide mpi functionalies for python**:
   ```sh
   pip install mpi4py
   
4. **Create a common shared folder on all of your systems and paste all the code files over there.**

## Usage

To run your code, follow these steps:

1. **To run the code on a single machine, use this command**:
   ```sh
   mpiexec -n num_cores python file_name.py
   
2. **To run the code on multiple machines, first create a hostfile with IP Addresses of your systems**:
   ```bash
   192.168.1.1 slots=4  # Master machine with 4 cores
   192.168.1.2 slots=3  # Slave machine 1 with 3 cores
   192.168.1.3 slots=3  # Slave machine 2 with 3 cores
   
3. **Finally run this command to start execution**:
   ```sh
   mpiexec --hostfile hosts -np <total_cores> python file_name.py
4. **Replace file_name.py with any of the following:**
   It_gr.py: Iterated Greedy Algorithm
   gen.py  : Genetic Analysis Algorithm
   pso.py  : Paricle Swarm Optimization Algorithm

## Performance Evaluation

The application was tested on 2 datasets:
1. 10,000 jobs across 10 machines
2. 1,000 jobs across 3 machines
   
**Performance comparisons show significant improvements in the distributed setup over the single-computer setup for the 1st datasets which is large, but single-computer setup outperforms the distributed setup when it comes to the 2nd dataset which is smaller, most likely due to context switching and other distributed overheads.**

