import numpy as np
import random
import parameters

def repair_chromosome(chromosome, job_info):
    def repair_sequence(sequence):
        last_index = 0
        for item in job_info.values():
            sequence[last_index: last_index + len(item)] = sorted(sequence[last_index: last_index + len(item)])
            last_index = last_index + len(item)

        return sequence
    def repair_assignment(assignment):
        curr_ind = 0
        for items in job_info.values():
            for item in items:
                machines = list(item.keys())
                if assignment[curr_ind] not in machines:
                    assignment[curr_ind] = random.choice(machines)
                curr_ind = curr_ind + 1

        return assignment

    new_chromosome = {'assignment': repair_assignment(chromosome['assignment']), 'sequence': repair_sequence(chromosome['sequence'])}
    return new_chromosome

def generate_initial_population_v2(jobs_operations_dict, num_machines):
    """
    Generates an initial population for the genetic algorithm considering varied number of operations per job.

    Args:
    - pop_size: Size of the population to generate.
    - jobs_operations_dict: A dictionary where keys are job IDs and values are the number of operations for each job.
    - num_machines: The number of machines available.

    Returns:
    - A list of dictionaries representing the initial population. Each dictionary has two keys: 'assignment' and 'sequence',
      where 'assignment' is a list of machine assignments for each operation, and 'sequence' is a list of operations in the
      order they are to be processed.
    """
    num_operations = 0  # Total number of operations across all jobs
    for items in jobs_operations_dict.values():
        num_operations = num_operations + len(items)

    # Initialize the population as a list
    population = []

    pop_size =  parameters.POPULATION_SIZE

    # Fill in the population
    for _ in range(pop_size):
        chromosome = {
            'assignment': np.random.randint(1, num_machines + 1, size=num_operations).tolist(),
            'sequence': np.random.permutation(np.arange(1, num_operations + 1)).tolist()
        }
        chromosome = repair_chromosome(chromosome=chromosome, job_info=jobs_operations_dict)
        chromosome['fitness'] = calculate_fitness(chromosome=chromosome, job_info=jobs_operations_dict)
        population.append(chromosome)

    return population


def calculate_fitness(chromosome, job_info):
    time_taken = 0 # fitness
    assignment = chromosome['assignment']
    sequence = chromosome['sequence']
    no_machines = 2

    machine_completion_time = {i + 1: 0 for i in range(no_machines)}
    operation_completion_time = {}

    rank_seq = []
    for index in range(len(sequence)):
        rank_seq.append((sequence[index], index))

    rank_seq.sort()

    operations = []
    for key, value in job_info.items():
        no_operations = len(value)
        for ind in range(no_operations):
            operations.append((key, ind))

    for ind in range(0, len(rank_seq)):
        # Calculating the machining time
        machining_time = job_info[operations[rank_seq[ind][1]][0]][operations[rank_seq[ind][1]][1]][assignment[rank_seq[ind][1]]]

        #Calculating the prev operation completion time
        prev_operation_time = 0
        if operations[rank_seq[ind][1]][1] != 0:
            prev_operation_time = operation_completion_time[operations[rank_seq[ind][1] - 1]]

        # Calculating the current machine ending time
        curr_machine_time = machine_completion_time[assignment[rank_seq[ind][1]]]

        # Calculating the final operation completion time
        curr_operation_completion_time = max(prev_operation_time, curr_machine_time) + machining_time

        # Updating the value of current operation
        operation_completion_time[operations[rank_seq[ind][1]]] = curr_operation_completion_time

        # Updating the current machine time
        machine_completion_time[assignment[rank_seq[ind][1]]] = curr_operation_completion_time

        # Assigning the max completion time to time_taken
        time_taken = max(time_taken, curr_operation_completion_time)

    return time_taken


def elitist_selection(population):
    """
    Selects the top-performing individuals from the population based on their fitness values using elitism.

    Args:
    - population: A list of dictionaries, where each dictionary represents a chromosome and contains
                  'assignment', 'sequence', and 'fitness' keys.
    - num_elites: The number of elite individuals to select for the next generation.

    Returns:
    - A list of dictionaries representing the elite chromosomes.
    """
    # Sort the population by the fitness value in ascending order (since lower fitness is better)
    population = sorted(population, key=lambda x: x['fitness'])

    # Select the top N elites
    num_elites = parameters.POPULATION_SIZE
    elites = population[:num_elites]

    return elites

# Example usage
jobs_operations_dict = {'J1': [{1: 2}, {1: 3, 2: 2}],
                        'J2': [{2: 1}, {1: 3}, {1: 4, 2: 2}],
                        'J3': [{1: 3, 2: 1}],
                        }
num_machines = 2

# Generate the initial population
initial_population_v2 = generate_initial_population_v2(jobs_operations_dict, num_machines)
print('Initial Population: ')
for i, chromosome in enumerate(initial_population_v2, start=1):
    print(f"Chromosome {i}: {chromosome}")

# Test Chromosome
# chromosome = {'assignment': [1, 1, 2, 1, 2, 1], 'sequence': [1, 2, 3, 4, 5, 6]}
# print(calculate_fitness(chromosome=chromosome, job_info=jobs_operations_dict))