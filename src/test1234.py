import random
import numpy as np

# Function to initialize the population with random binary strings
def initialize_population(size):
    return ["".join(random.choice("01") for _ in range(11)) for _ in range(size)]

# Fitness function: count the number of correct predictions
def fitness(individual, dataset):
    score = sum(1 for x, y in dataset if x[:-1] == individual[:-1] and int(y) == int(individual[-1]))
    return score / len(dataset) if len(dataset) > 0 else 0

# Selection using Roulette Wheel Selection
def selection(population, dataset):
    fitness_values = [fitness(ind, dataset) for ind in population]
    total_fit = sum(fitness_values)
    
    if total_fit == 0:
        return random.choice(population)
    
    probabilities = [f / total_fit for f in fitness_values]
    return np.random.choice(population, p=probabilities)

# One-point crossover function
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

# Mutation function
def mutate(individual, mutation_rate):
    individual = list(individual)
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = "1" if individual[i] == "0" else "0"
    return "".join(individual)

# Genetic Algorithm execution with parameters p, r, m
def genetic_algorithm(p, r, m, generations, dataset):
    population = initialize_population(p)

    for gen in range(generations):
        new_population = []
        num_replacements = int(r * p)  # Number of individuals to replace each generation

        for _ in range(num_replacements // 2):
            parent1, parent2 = selection(population, dataset), selection(population, dataset)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1, m), mutate(child2, m)])

        # Retain the top (1 - r) * p individuals from the previous generation
        survivors = sorted(population, key=lambda x: fitness(x, dataset), reverse=True)[: p - num_replacements]
        population = survivors + new_population

        # Get the best individual of this generation
        best_individual = max(population, key=lambda x: fitness(x, dataset))
    
        if fitness(best_individual, dataset) == 1.0:  # Stop early if perfect accuracy is achieved
            break

    return max(population, key=lambda x: fitness(x, dataset))

# PlayTennis dataset (expanded with more samples)
dataset = [
    ("10101010011", 1), ("11010011000", 0), ("01101100101", 1), ("10111010100", 0),
    ("00011100010", 1), ("11100010111", 0), ("10011001101", 1), ("01010101010", 0),
    ("11001110001", 1), ("00110011100", 0), ("01100010010", 1), ("10101101110", 0),
    ("10010100100", 1), ("11011000011", 0), ("01110101001", 1), ("00001011000", 0),
    ("11110101101", 1), ("10001010111", 0), ("00101100011", 1), ("11000110110", 0)
]

# Running experiments with different parameters
experiments = [
    {"p": 4, "r": 0.5, "m": 0.05},
    {"p": 20, "r": 0.3, "m": 0.01},
    {"p": 50, "r": 0.4, "m": 0.1},
]

print("\n### Running experiments with different parameters ###")
for exp in experiments:
    print(f"\nRunning Genetic Algorithm with Population Size = {exp['p']}, Replacement Rate = {exp['r']}, Mutation Rate = {exp['m']}")
    best_rule = genetic_algorithm(exp['p'], exp['r'], exp['m'], generations=50, dataset=dataset)
    print("Final Best Hypothesis:", best_rule)
