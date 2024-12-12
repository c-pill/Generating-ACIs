# Greedy Solution for Generating Accurately Censored Images

from Container import Greedy_Solution
import os
import time
import winsound as ws


img = "Me.png"
goal_percentage = 50

# used to run GSGACI back to back
runs = 1

print("\nGSGACI started...")
print("-------------")

for r in range(runs):
    start_time = time.perf_counter()
    greedy = Greedy_Solution(f"./images/{img}", goal_percentage)
    greedy.greedy_generate()
    greedy.evaluate_fitness()
    print(f"Run complete in {time.perf_counter()-start_time:.3f} seconds")
    print(f"Fit: {greedy.fit}")
    greedy.display_result()
    # greedy.create_folder_and_save("GSGACI")
    print("-----------------")
    print(f"Censored images stored in {greedy.path}")
    print("------------------")
    print(f"GSGACI completed")
    # small gap between runs so pseudo random numbers aren't repeated
    time.sleep(.25)

path = os.path.realpath(greedy.path)
os.startfile(path)

# sound to alert user when finished
ws.Beep(700, 1000)