# evo_toy_improved.py - Improved evolutionary strategy (educational)
# Run with: python evo_toy_improved.py
import random
import math
import statistics

# A more robust random number generator for better statistical properties
from numpy.random import normal, uniform, choice
import numpy as np

# Target function we want to "discover" (known to us)
def target_function(x):
    """
    Multimodal but safe: sum of sines + parabola.
    """
    return sum(math.sin(v) for v in x) - 0.05 * sum(v*v for v in x)

# Individual represented as a numpy array for efficiency
def random_individual(dim, scale=2.0):
    return uniform(-scale, scale, dim)

def mutate(ind, sigma):
    """
    Standard Gaussian mutation.
    """
    return ind + normal(0, sigma, ind.shape)

def crossover(a, b):
    """
    Uniform Crossover, more standard for real-valued problems.
    """
    return np.where(choice([True, False], size=a.shape), a, b)

def evaluate_population(pop):
    """
    Vectorized evaluation for efficiency.
    """
    return [target_function(ind) for ind in pop]

def evolve(dim=8, pop_size=80, generations=400,
           elite_frac=0.1, mutation_sigma_start=0.5, mutation_sigma_end=0.05):
    """
    Evolves a population to optimize the target function.
    """
    # init
    population = [random_individual(dim) for _ in range(pop_size)]
    history = []
    
    for gen in range(generations):
        # Adaptive mutation sigma
        sigma = mutation_sigma_start - (gen / generations) * (mutation_sigma_start - mutation_sigma_end)

        scores = evaluate_population(population)
        
        # Sort population and scores together
        ranked_pop = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        ranked_scores = [s for s, _ in ranked_pop]
        ranked_population = [p for _, p in ranked_pop]

        best_score = ranked_scores[0]
        avg_score = statistics.mean(ranked_scores)
        
        history.append((gen, best_score, avg_score))

        if gen % 40 == 0 or gen == generations-1:
            print(f"Gen {gen:3d}: best={best_score:.4f}, avg={avg_score:.4f}, sigma={sigma:.4f}")

        # Elitism
        elite_count = max(2, int(pop_size * elite_frac))
        elites = ranked_population[:elite_count]

        # Generate new population
        new_pop = elites[:]
        while len(new_pop) < pop_size:
            # Tournament selection to pick parents
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)
            
            # Create a child via crossover and mutation
            child = crossover(parent1, parent2)
            child = mutate(child, sigma=sigma)
            new_pop.append(child)
        
        population = new_pop
    
    return population, history

if __name__ == "__main__":
    final_pop, hist = evolve()
    best_ind = max(final_pop, key=target_function)
    print("\nFinal best individual:", best_ind)
    print("Final best score:", target_function(best_ind))
