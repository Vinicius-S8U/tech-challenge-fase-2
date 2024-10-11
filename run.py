import random
import streamlit as st
import time
import matplotlib.pyplot as plt

num_items = 100
item_values = [random.randint(10, 1000) for _ in range(num_items)]
item_weights = [random.randint(5, 500) for _ in range(num_items)]
inventory_capacity = 20000

# Function to calculate the fitness of an individual solution
def calculate_fitness(solution,item_values,item_weights,inventory_capacity):
    total_value = 0
    total_weight = 0
    for i in range(len(solution)):
        if solution[i] == 1:
            total_value += item_values[i]
            total_weight += item_weights[i]
        if total_weight > inventory_capacity:
            return 0, total_weight
    return total_value, total_weight

# Generate a random population of possible solutions
def generate_random_population(size, length):
    return [[random.randint(0, 1) for _ in range(length)] for _ in range(size)]

# Perform crossover
def crossover(parent1, parent2):
    random_index_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:random_index_point] + parent2[random_index_point:]
    child2 = parent2[:random_index_point] + parent1[random_index_point:]
    return child1, child2

# Perform mutation
def mutate(solution, mutation_rate):
    mutated_solution = list(solution)
    for i in range(len(solution)):
        if random.random() > mutation_rate:
            mutated_solution[i] = 1 if mutated_solution[i] == 0 else 0
    return mutated_solution

# Genetic algorithm loop
def genetic_algorithm(population_size, num_generations, mutation_rate,num_items,isElitism):
 
    population = generate_random_population(population_size, num_items)
    best_fitness_values = []
    
    # Create an empty chart
    chart = st.empty()
    
    for generation in range(num_generations):
        # Sort population based on fitness
        population.sort(key=lambda x: calculate_fitness(x,item_values,item_weights,inventory_capacity)[0], reverse=True)
        
        # Get best solution from population
        best_fitness = calculate_fitness(population[0],item_values,item_weights,inventory_capacity)
        best_fitness_values.append(best_fitness)
        
        # Update the chart dynamically every generation
        chart.line_chart(best_fitness_values)  
        
        if(isElitism):
         new_population = [population[0]]  # Keep the best individual (elitism)
        else:
            new_population = []
        
        # Generate new population via crossover and mutation
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population[:10], k=2)  # Selection based on truncate
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        
        population = new_population
        
        time.sleep(0.01)  # Add a small delay to visualize real-time updates
        
    return best_fitness_values

# Streamlit app
st.title("Genetic Algorithm - Knapsack Problem")

# Sidebar for parameter adjustment
# num_items = st.sidebar.slider("Num Itens", 10, 500, 200)
population_size = st.sidebar.slider("Population Size", 10, 100, 20)
num_generations = st.sidebar.slider("Number of Generations", 10, 1000, 100)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 1.0, 0.01)
isElitism = st.sidebar.checkbox("Elitism")

# Run the genetic algorithm and display progress
if st.button("Run Algorithm"):
    best_fitness_values = genetic_algorithm(population_size, num_generations, mutation_rate,num_items,isElitism)
    # Plot results
    st.write("Best Fitness by Generation")
    fig, ax = plt.subplots()
    ax.plot(range(len(best_fitness_values)), best_fitness_values, marker="o")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Best Fitness Over Generations")
    st.pyplot(fig)
    st.write(f"Best Fitness (Value,Weight): {best_fitness_values[-1] }")