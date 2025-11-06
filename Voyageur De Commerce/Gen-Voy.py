import random

class VoyageurGenetic:
    def __init__(self, matrice_distances):
        self.matrice_distances = matrice_distances
        self.nombre_villes = len(matrice_distances)

    def calculer_distance(self, solution):
        distance = 0
        for i in range(len(solution)):
            ville_actuelle = solution[i]
            ville_suivante = solution[(i + 1) % len(solution)]
            distance += self.matrice_distances[ville_actuelle][ville_suivante]
        return distance

    def generer_individu(self):
        individu = list(range(self.nombre_villes))
        random.shuffle(individu)
        return individu

    def initialiser_population(self, taille_population):
        return [self.generer_individu() for _ in range(taille_population)]

    def evaluer_fitness(self, individu):
        distance = self.calculer_distance(individu)
        return 1 / distance

    def selection_roulette(self, population, fitnesses):
        total_fitness = sum(fitnesses)
        probabilites = [f / total_fitness for f in fitnesses]
        return random.choices(population, weights=probabilites, k=2)

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

    def mutation_echange(self, individu):
        mutant = individu.copy()
        i, j = random.sample(range(self.nombre_villes), 2)
        mutant[i], mutant[j] = mutant[j], mutant[i]
        return mutant

    def algorithme_genetique_roulette(self, population_size=50, generations=100, mutation_rate=0.1, crossover_type='simple'):
        population = self.initialiser_population(population_size)
        best_individual = None
        best_distance = float('inf')

        for generation in range(generations):
            fitnesses = [self.evaluer_fitness(ind) for ind in population]
            new_population = []

            for _ in range(population_size // 2):
                parent1, parent2 = self.selection_roulette(population, fitnesses)

                if crossover_type == 'simple':
                    child1, child2 = self.crossover_simple(parent1, parent2)
                elif crossover_type == 'double':
                    child1, child2 = self.crossover_double(parent1, parent2)
                elif crossover_type == 'uniform':
                    child1, child2 = self.crossover_uniform(parent1, parent2)
                else:
                    child1, child2 = self.crossover_simple(parent1, parent2)

                if random.random() < mutation_rate:
                    child1 = self.mutation_echange(child1)
                if random.random() < mutation_rate:
                    child2 = self.mutation_echange(child2)

                new_population.extend([child1, child2])

            population = new_population
            current_best_idx = fitnesses.index(max(fitnesses))
            current_best = population[current_best_idx]
            current_distance = self.calculer_distance(current_best)

            if current_distance < best_distance:
                best_individual = current_best
                best_distance = current_distance

        return best_individual, best_distance

    def algorithme_genetique_rang(self, population_size=50, generations=100, mutation_rate=0.1, crossover_type='simple'):
        population = self.initialiser_population(population_size)
        best_individual = None
        best_distance = float('inf')

        for generation in range(generations):
            fitnesses = [self.evaluer_fitness(ind) for ind in population]
            new_population = []

            for _ in range(population_size // 2):
                parent1, parent2 = self.selection_rang(population, fitnesses)

                if crossover_type == 'simple':
                    child1, child2 = self.crossover_simple(parent1, parent2)
                elif crossover_type == 'double':
                    child1, child2 = self.crossover_double(parent1, parent2)
                elif crossover_type == 'uniform':
                    child1, child2 = self.crossover_uniform(parent1, parent2)
                else:
                    child1, child2 = self.crossover_simple(parent1, parent2)

                if random.random() < mutation_rate:
                    child1 = self.mutation_echange(child1)
                if random.random() < mutation_rate:
                    child2 = self.mutation_echange(child2)

                new_population.extend([child1, child2])

            population = new_population
            current_best_idx = fitnesses.index(max(fitnesses))
            current_best = population[current_best_idx]
            current_distance = self.calculer_distance(current_best)

            if current_distance < best_distance:
                best_individual = current_best
                best_distance = current_distance

        return best_individual, best_distance


# Matrice des distances
matrice_distances = [
    [0, 2, 2, 7, 15, 2, 5, 7, 6, 5],
    [2, 0, 10, 4, 7, 3, 7, 15, 8, 2],
    [2, 10, 0, 1, 4, 3, 3, 4, 2, 3],
    [7, 4, 1, 0, 2, 15, 7, 7, 5, 4],
    [7, 10, 4, 2, 0, 7, 3, 2, 2, 7],
    [2, 3, 3, 7, 7, 0, 1, 7, 2, 10],
    [5, 7, 3, 7, 3, 1, 0, 2, 1, 3],
    [7, 7, 4, 7, 2, 7, 2, 0, 1, 10],
    [6, 8, 2, 5, 2, 2, 1, 1, 0, 15],
    [5, 2, 3, 4, 7, 10, 3, 10, 15, 0]
]

voyageur_genetic = VoyageurGenetic(matrice_distances)

crossover_types = ['simple', 'double', 'uniform']

print("=== SELECTION PAR ROULETTE ===")
for crossover_type in crossover_types:
    best_solution, best_distance = voyageur_genetic.algorithme_genetique_roulette(
        population_size=50,
        generations=100,
        crossover_type=crossover_type
    )

    print(f"\n{crossover_type.upper()} CROSSOVER - Best solution found:")
    print(f"Distance minimum: {best_distance}")
    print("Sequence:", best_solution)

print("\n=== SELECTION PAR RANG ===")
for crossover_type in crossover_types:
    best_solution, best_distance = voyageur_genetic.algorithme_genetique_rang(
        population_size=50,
        generations=100,
        crossover_type=crossover_type
    )

    print(f"\n{crossover_type.upper()} CROSSOVER - Best solution found:")
    print(f"Distance minimum: {best_distance}")
    print("Sequence:", best_solution)