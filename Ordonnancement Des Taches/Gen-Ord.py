import random


class JobShopSchedulingGenetic:
    def __init__(self, jobs):
        self.jobs = jobs
        self.num_machines = max(machine for job in jobs.values() for machine, _ in job) + 1
        self.num_jobs = len(jobs)

    def decode_schedule(self, operation_sequence):
        machine_times = [0] * self.num_machines
        job_times = [0] * (self.num_jobs + 1)
        schedule = []
        makespan = 0

        for job_id, op_num in operation_sequence:
            machine, duration = self.jobs[job_id][op_num]
            start_time = max(machine_times[machine], job_times[job_id])
            end_time = start_time + duration
            machine_times[machine] = end_time
            job_times[job_id] = end_time

            schedule.append({
                'job': job_id, 'operation': op_num, 'machine': machine,
                'start': start_time, 'end': end_time, 'duration': duration
            })
            makespan = max(makespan, end_time)

        return schedule, makespan

    def generate_individual(self):
        operations = []
        for job_id in self.jobs:
            for op_num in range(len(self.jobs[job_id])):
                operations.append((job_id, op_num))
        random.shuffle(operations)
        return operations

    def initialize_population(self, population_size):
        return [self.generate_individual() for _ in range(population_size)]

    def evaluate_fitness(self, individual):
        _, makespan = self.decode_schedule(individual)
        return 1 / makespan

    def selection_roulette(self, population, fitnesses):
        total_fitness = sum(fitnesses)
        probabilities = [f / total_fitness for f in fitnesses]
        parent1 = random.choices(population, weights=probabilities)[0]
        parent2 = random.choices(population, weights=probabilities)[0]
        return parent1, parent2

    def selection_rang(self, population, fitnesses):
        sorted_population = [x for _, x in sorted(zip(fitnesses, population), reverse=True)]
        ranks = list(range(1, len(population) + 1))
        total_ranks = sum(ranks)
        probabilities = [rank / total_ranks for rank in ranks]
        return random.choices(sorted_population, weights=probabilities, k=2)

    def crossover_simple(self, parent1, parent2):
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + [gene for gene in parent2 if gene not in parent1[:point]]
        child2 = parent2[:point] + [gene for gene in parent1 if gene not in parent2[:point]]
        return child1, child2

    def crossover_double(self, parent1, parent2):
        point1 = random.randint(1, len(parent1) - 2)
        point2 = random.randint(point1 + 1, len(parent1) - 1)

        segment1 = parent1[point1:point2]
        child1 = []
        for gene in parent2:
            if len(child1) == point1:
                child1.extend(segment1)
            if gene not in segment1:
                child1.append(gene)

        segment2 = parent2[point1:point2]
        child2 = []
        for gene in parent1:
            if len(child2) == point1:
                child2.extend(segment2)
            if gene not in segment2:
                child2.append(gene)

        return child1[:len(parent1)], child2[:len(parent2)]

    def crossover_uniform(self, parent1, parent2):
        child1 = []
        child2 = []

        all_genes = set(parent1 + parent2)

        for i in range(len(parent1)):
            if random.random() < 0.5:
                if parent1[i] not in child1:
                    child1.append(parent1[i])
                if parent2[i] not in child2:
                    child2.append(parent2[i])
            else:
                if parent2[i] not in child1:
                    child1.append(parent2[i])
                if parent1[i] not in child2:
                    child2.append(parent1[i])

        for gene in all_genes:
            if gene not in child1:
                child1.append(gene)
            if gene not in child2:
                child2.append(gene)

        return child1[:len(parent1)], child2[:len(parent2)]

    def mutation(self, individual):
        mutated = individual.copy()
        machine_ops = {}
        for i, (job_id, op_num) in enumerate(mutated):
            machine = self.jobs[job_id][op_num][0]
            if machine not in machine_ops:
                machine_ops[machine] = []
            machine_ops[machine].append(i)

        valid_machines = [m for m in machine_ops if len(machine_ops[m]) >= 2]
        if valid_machines:
            machine = random.choice(valid_machines)
            idx1, idx2 = random.sample(machine_ops[machine], 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return mutated

    def algorithme_genetique_roulette(self, population_size=10, generations=100, mutation_rate=0.1, crossover_type='simple'):
        population = self.initialize_population(population_size)
        best_individual = None
        best_makespan = float('inf')
        best_schedule = None

        for generation in range(generations):
            fitnesses = [self.evaluate_fitness(ind) for ind in population]
            new_population = []

            for _ in range(population_size // 2):
                parent1, parent2 = self.selection_roulette(population, fitnesses)
                child1, child2 = self.crossover_simple(parent1, parent2) if crossover_type == 'simple' else \
                                self.crossover_double(parent1, parent2) if crossover_type == 'double' else \
                                self.crossover_uniform(parent1, parent2)

                if random.random() < mutation_rate:
                    child1 = self.mutation(child1)
                if random.random() < mutation_rate:
                    child2 = self.mutation(child2)

                new_population.extend([child1, child2])

            population = new_population
            current_best_idx = fitnesses.index(max(fitnesses))
            current_best = population[current_best_idx]
            current_schedule, current_makespan = self.decode_schedule(current_best)

            if current_makespan < best_makespan:
                best_individual = current_best
                best_schedule = current_schedule
                best_makespan = current_makespan

        return best_individual, best_schedule, best_makespan

    def algorithme_genetique_rang(self, population_size=10, generations=100, mutation_rate=0.1, crossover_type='simple'):
        population = self.initialize_population(population_size)
        best_individual = None
        best_makespan = float('inf')
        best_schedule = None

        for generation in range(generations):
            fitnesses = [self.evaluate_fitness(ind) for ind in population]
            new_population = []

            for _ in range(population_size // 2):
                parent1, parent2 = self.selection_rang(population, fitnesses)
                child1, child2 = self.crossover_simple(parent1, parent2) if crossover_type == 'simple' else \
                                self.crossover_double(parent1, parent2) if crossover_type == 'double' else \
                                self.crossover_uniform(parent1, parent2)

                if random.random() < mutation_rate:
                    child1 = self.mutation(child1)
                if random.random() < mutation_rate:
                    child2 = self.mutation(child2)

                new_population.extend([child1, child2])

            population = new_population
            current_best_idx = fitnesses.index(max(fitnesses))
            current_best = population[current_best_idx]
            current_schedule, current_makespan = self.decode_schedule(current_best)

            if current_makespan < best_makespan:
                best_individual = current_best
                best_schedule = current_schedule
                best_makespan = current_makespan

        return best_individual, best_schedule, best_makespan


# Example usage with all crossover types and both selection methods
if __name__ == "__main__":
    jobs = {
        1: [(0, 3), (1, 2), (2, 2)],
        2: [(0, 2), (2, 1), (1, 4)],
        3: [(1, 4), (0, 3), (2, 1)],
    }

    job_shop_genetic = JobShopSchedulingGenetic(jobs)

    crossover_types = ['simple', 'double', 'uniform']

    print("=== SELECTION PAR ROULETTE ===")
    for crossover_type in crossover_types:
        best_solution, best_schedule, best_makespan = job_shop_genetic.algorithme_genetique_roulette(
            population_size=10,
            generations=50,
            crossover_type=crossover_type
        )

        print(f"\n{crossover_type.upper()} CROSSOVER - Best solution found:")
        print(f"Makespan minimum: {best_makespan}")
        print("Sequence:", best_solution)

    print("\n=== SELECTION PAR RANG ===")
    for crossover_type in crossover_types:
        best_solution, best_schedule, best_makespan = job_shop_genetic.algorithme_genetique_rang(
            population_size=10,
            generations=50,
            crossover_type=crossover_type
        )

        print(f"\n{crossover_type.upper()} CROSSOVER - Best solution found:")
        print(f"Makespan minimum: {best_makespan}")
        print("Sequence:", best_solution)