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
    new_population = sorted(population, key=lambda x: x['fitness'])

    # Select the top N elites
    num_elites = parameters.POPULATION_SIZE
    elites = new_population[:num_elites]

    return elites


def crossover(parent1, parent2, job_info):
    def one_point_crossover():
        """
        Performs a one-point crossover between two parent chromosomes.

        Args:
        - parent1, parent2: Parent chromosomes, each a dictionary with 'assignment' and 'sequence'.

        Returns:
        - Two offspring chromosomes as dictionaries with 'assignment' and 'sequence'.
        """
        # Ensure the crossover point is within the range of the chromosomes' lengths
        crossover_point = random.randint(1, len(parent1['assignment']) - 2)

        # Create offspring by swapping subsequences after the crossover point
        offspring1_assignment = parent1['assignment'][:crossover_point] + parent2['assignment'][crossover_point:]
        offspring2_assignment = parent2['assignment'][:crossover_point] + parent1['assignment'][crossover_point:]

        return offspring1_assignment, offspring2_assignment

    def partial_matched_crossover():
        """
        Performs a Partial Matched Crossover (PMX) on the sequence part of two parent chromosomes.

        Args:
        - parent1, parent2: Parent chromosomes, each a dictionary with 'sequence'.

        Returns:
        - Two offspring chromosomes as dictionaries with 'sequence' only.
        """
        # Determine the crossover points
        size = len(parent1['sequence'])
        cx_point1, cx_point2 = sorted(random.sample(range(size), 2))

        # Create offspring sequence placeholders
        offspring1_seq = [None] * size
        offspring2_seq = [None] * size

        # Copy the segment between crossover points from each parent to each offspring
        for i in range(cx_point1, cx_point2 + 1):
            offspring1_seq[i] = parent2['sequence'][i]
            offspring2_seq[i] = parent1['sequence'][i]

        # Fill in the remaining positions with the elements from the respective parent,
        # ensuring that no duplicates occur.
        for i in range(size):
            if not offspring1_seq[i]:
                if parent1['sequence'][i] not in offspring1_seq:
                    offspring1_seq[i] = parent1['sequence'][i]
                else:
                    # Find the element in parent2 that corresponds to this position and use it if not already in offspring
                    for j in range(size):
                        if parent2['sequence'][j] == parent1['sequence'][i]:
                            mapping_element = parent1['sequence'][j]
                            if mapping_element not in offspring1_seq:
                                offspring1_seq[i] = mapping_element
                                break
            if not offspring2_seq[i]:
                if parent2['sequence'][i] not in offspring2_seq:
                    offspring2_seq[i] = parent2['sequence'][i]
                else:
                    for j in range(size):
                        if parent1['sequence'][j] == parent2['sequence'][i]:
                            mapping_element = parent2['sequence'][j]
                            if mapping_element not in offspring2_seq:
                                offspring2_seq[i] = mapping_element
                                break

        # Handle any None values by assigning missing elements (this could happen in edge cases)
        remaining_elements1 = [item for item in parent1['sequence'] if item not in offspring1_seq]
        remaining_elements2 = [item for item in parent2['sequence'] if item not in offspring2_seq]
        for i in range(size):
            if offspring1_seq[i] is None:
                offspring1_seq[i] = remaining_elements1.pop(0)
            if offspring2_seq[i] is None:
                offspring2_seq[i] = remaining_elements2.pop(0)

        return offspring1_seq, offspring2_seq

    offspring1_assignment, offspring2_assignment = one_point_crossover()
    offspring1_seq, offspring2_seq = partial_matched_crossover()

    offspring1 = {'assignment': offspring1_assignment, 'sequence': offspring1_seq}
    offspring1 = repair_chromosome(offspring1, job_info=job_info)
    offspring1['fitness'] = calculate_fitness(offspring1, job_info=job_info)

    offspring2 = {'assignment': offspring2_assignment, 'sequence': offspring2_seq}
    offspring2 = repair_chromosome(offspring2, job_info=job_info)
    offspring2['fitness'] = calculate_fitness(offspring2, job_info=job_info)

    return offspring1, offspring2

def mutate(chromosome, job_info, num_machines):
    mutation_type = random.choice(['assignment', 'sequence'])
    mutated_chromosome = chromosome

    if mutation_type == 'assignment':
        # Randomly choose an operation and assign it to a new machine
        op_index = random.randint(0, len(chromosome['assignment']) - 1)
        mutated_chromosome['assignment'][op_index] = random.randint(1, num_machines)
    else:
        # Swap two operations in the sequence
        i, j = random.sample(range(len(chromosome['sequence'])), 2)
        mutated_chromosome['sequence'][i], mutated_chromosome['sequence'][j] = mutated_chromosome['sequence'][j], mutated_chromosome['sequence'][i]

    mutated_chromosome = repair_chromosome(chromosome, job_info)
    mutated_chromosome['fitness'] = calculate_fitness(chromosome=chromosome, job_info=job_info)

    return mutated_chromosome


def genetic_algorithm(job_info, num_machines):
    """
    Executes the genetic algorithm for job shop scheduling.

    Args:
    - jobs_operations_dict: A dictionary mapping jobs to their number of operations.
    - num_machines: The number of machines available.
    - pop_size: The size of the population.
    - crossover_rate: The probability of crossover.
    - mutation_rate: The probability of mutation.
    - termination_criterion: The number of generations to run the algorithm for.

    Returns:
    - The best chromosome (solution) found and its fitness.
    """

    pop_size = parameters.POPULATION_SIZE
    jobs_operations_dict = job_info
    crossover_rate = parameters.CROSSOVER_RATE
    mutation_rate = parameters.MUTATION_RATE
    termination_criterion = parameters.TERMINATION_GENERATION


    # Generate the initial population
    population = generate_initial_population_v2(jobs_operations_dict, num_machines)

    # Main GA loop
    for generation in range(termination_criterion):
        # Selection
        population = elitist_selection(population)  # Example: keeping 2 elites

        # Crossover
        for ind in range(0, len(population) - 1, 2):
            if random.random() < crossover_rate:
                parent1, parent2 = population[ind], population[ind + 1]
                offspring1, offspring2 = crossover(parent1, parent2, job_info)
                population += [offspring1, offspring2]

        # Mutation
        mutated_pop = []
        for chromosome in population:
            if random.random() < mutation_rate:
                mutated_chromosome = mutate(chromosome, job_info, num_machines)
                mutated_pop.append(mutated_chromosome)

        population += mutated_pop

    # Find the best solution
    best_chromosome = min(population, key=lambda x: x['fitness'])
    return best_chromosome, best_chromosome['fitness']



# Example usage
jobs_operations_dict = {'J1': [{1: 2}, {1: 3, 2: 2}],
                        'J2': [{2: 1}, {1: 3}, {1: 4, 2: 2}],
                        'J3': [{1: 3, 2: 1}],
                        }
num_machines = 2

# Assuming operations_data and job_machine_constraints are defined
best_solution, best_fitness = genetic_algorithm(jobs_operations_dict, num_machines)
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)

# Generate the initial population
initial_population_v2 = generate_initial_population_v2(jobs_operations_dict, num_machines)
print('Initial Population: ')
for i, chromosome in enumerate(initial_population_v2, start=1):
    print(f"Chromosome {i}: {chromosome}")

print(crossover(initial_population_v2[0], initial_population_v2[1], jobs_operations_dict))

# Test Chromosome
# chromosome = {'assignment': [1, 1, 2, 1, 2, 1], 'sequence': [1, 2, 3, 4, 5, 6]}
# print(calculate_fitness(chromosome=chromosome, job_info=jobs_operations_dict))