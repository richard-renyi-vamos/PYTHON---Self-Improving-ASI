# evo_toy.py — Tiny evolutionary strategy (educational)
# Run with: python evo_toy.py
import random
import math
import statistics

# Target function we want to "discover" (known to us)
def target_function(x):
    # multimodal but safe: sum of sines + parabola
    return sum(math.sin(v) for v in x) - 0.05 * sum(v*v for v in x)

# Individual represented as a list of floats
def random_individual(dim, scale=2.0):
    return [random.uniform(-scale, scale) for _ in range(dim)]

def mutate(ind, sigma=0.2):
    return [v + random.gauss(0, sigma) for v in ind]

def crossover(a, b):
    return [(va if random.random() < 0.5 else vb) for va, vb in zip(a, b)]

def evaluate_population(pop):
    return [target_function(ind) for ind in pop]

def evolve(dim=5, pop_size=60, generations=200,
           elite_frac=0.2, mutation_sigma=0.15):
    # init
    population = [random_individual(dim) for _ in range(pop_size)]
    history = []
    for gen in range(generations):
        scores = evaluate_population(population)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_score = scores[best_idx]
        avg_score = statistics.mean(scores)
        history.append((gen, best_score, avg_score))
        if gen % 20 == 0 or gen == generations-1:
            print(f"Gen {gen:3d}: best={best_score:.4f}, avg={avg_score:.4f}")
        # selection
        ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        elite_count = max(2, int(pop_size * elite_frac))
        elites = [ind for _, ind in ranked[:elite_count]]
        # generate new population
        new_pop = elites[:]
        while len(new_pop) < pop_size:
            a, b = random.sample(elites, 2)
            child = crossover(a, b)
            child = mutate(child, sigma=mutation_sigma)
            new_pop.append(child)
        population = new_pop
    return population, history

if __name__ == "__main__":
    final_pop, hist = evolve(dim=8, pop_size=80, generations=400)
    best = max(final_pop, key=target_function)
    print("Best individual:", best)
    print("Best score:", target_function(best))
