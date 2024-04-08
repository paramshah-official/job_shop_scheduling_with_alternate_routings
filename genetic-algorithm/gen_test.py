import random

# Define the number of machines
num_machines = 4

# Define the job operations dictionary with machines as keys and time taken as values
jobs_operations_dict = {}

# Generate random job operations with varied number of operations and machines
num_jobs = 50
max_num_operations = 6
max_num_machines = 4

for job_idx in range(1, num_jobs + 1):
    num_operations = random.randint(1, max_num_operations)
    a = []
    for _ in range(num_operations):
        for x in range(num_machines):
            if random.random() > 0.5:
                operations = {}
                num_machines_for_operation = random.randint(1, max_num_machines)
                machines = random.sample(range(1, num_machines + 1), num_machines_for_operation)
                times = {machine: random.randint(1, 20) for machine in machines}
                operations.update(times)
                a.append(operations)
    jobs_operations_dict[f'J{job_idx}'] = a

print(jobs_operations_dict)