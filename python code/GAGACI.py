# Genetic Algorithm for Generating Accurately Censored Images

from Container import GA_Generation
import gc
import os
import time
import winsound as ws

# if image too large might crash program (too much memory used)
# if this is the case screen shot image and use that
# image must be a PNG file
img = "Face.png" 
population_size = 15
goal_percentage = 75
iterations      = 10

print("\nGAGACI started...")
print("-------------")

# create population container
parent = GA_Generation(f"./images/{img}", population_size, goal_percentage)

start_time = time.perf_counter()
parent.generate_population()
parent.evaluate_population()

for i in range(iterations):
    # selection & crossover
    parent.tournament_select(3)
    # mutate
    parent.mutate_children(i, iterations)
    # survivor selection
    parent.round_robin(7, 80)
    parent.sort_wins()
    parent.survivor_select()
    if i % 50 == 0: gc.collect()

print(f"Run complete in {time.perf_counter()-start_time:.3f} seconds")
# parent.create_folder_and_save("GAGACI")
print(f"Fitness {parent.best_fit}")
print(f"%: {goal_percentage}")
print("-----------------")
print(f"Censored images stored in {parent.path}")
print("------------------")
print(f"GAGACI completed")

# path = os.path.realpath(parent.path)
# os.startfile(path)

# sound to alert user when finished
ws.Beep(700, 1000)