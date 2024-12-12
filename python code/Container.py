from ctypes import *
from PIL import Image
import numpy as np
import os
import random as rd

results_folder = './alg results'
if not os.path.exists(results_folder): 
    os.mkdir(results_folder)

lib = CDLL(os.path.join("./c code", "clibrary.so"))

c_int_p = POINTER(c_uint)
c_float_p = POINTER(c_float)

lib.evaluate_fitness.restype = c_float
lib.mass_swap.restype = c_int_p
lib.smart_swap.restype = c_int_p
lib.greedy_generate.restype = c_int_p

rng = np.random.default_rng()

class Generation: 

    # path to results folder
    path = ' '

    # constructor takes original image's file, the number of individuals to be generated, 
    # the percentage of likeness desired, and the margin of error for 
    # accepting individuals as solutions
    def __init__(self, img_file, N, goal):
        # original image's file
        self.orig_img_file = img_file
        # original image
        self.orig_image = Image.open(img_file).convert('RGB')
        # width and height of all images
        self.width, self.height = self.orig_image.size
        # number of pixels in all images
        self.num_pixels = self.width * self.height
        # original image's pixels stored as a 1D array (2D if considering each
        # pixel's individual values)
        self.orig_pixels = np.asarray(self.orig_image).reshape((self.num_pixels, 3)).astype(c_uint)
        # original image's pixels converted to a flattened c_int_p
        self.orig_pixels_p = self.orig_pixels.flatten().ctypes.data_as(c_int_p)
        # number of individuals in a population
        self.size = N
        # percentage of likeness desired
        self.goal = goal

    # show the original image
    def display_original(self):
        self.orig_image.show()

    # show best individual
    def display_best(self):
        Image.fromarray(np.reshape(self.best_ind, (self.height, self.width, 3)).astype(np.uint8)).show()

    # show a generated individual from population as an images
    def display_individual(self, ind_pos):
        Image.fromarray(np.reshape(self.population[ind_pos], (self.height, self.width, 3)).astype(np.uint8)).show()

    # shows all generated individuals in population as images
    def display_population(self):
        for i in range(len(self.population)): self.display_individual(i)

    # prints individual using lib
    def print_individual(self, ind_pos):
        lib.print_individual(self.population[ind_pos].flatten().ctypes.data_as(c_int_p), self.num_pixels)
    
    # prints original using lib
    def print_original(self):
        lib.print_individual(self.orig_pixels_p, self.num_pixels)  
    
    # generates N number of individuals for initial population
    # uses mass_swap_mutate to randomly scramble the initial population
    def generate_population(self):
        # stores best individual
        self.best_ind = []
        # stores fitness of best individual
        self.best_fit = 100
        # population of images in the current generation
        self.population = []
        for i in range(self.size):
            self.population.append(self.orig_pixels)
            self.population[i] = self.mass_swap_mutate(i)

    # evaluates the fitness of an individual in the population using lib
    # fitness is how close the percentage of likeness an image (to the original)
    # is to the goal percentage of likeness
    def evaluate_fitness(self, ind_pos):
        fit = lib.evaluate_fitness(self.population[ind_pos].flatten().ctypes.data_as(c_int_p), self.orig_pixels_p, c_double(self.goal), self.num_pixels)
        if fit < self.best_fit:
            self.best_ind = self.population[ind_pos]
            self.best_fit = fit
        self.fitness.append(fit)

    # evaluates the fitness of each individual in the population using lib
    def evaluate_population(self):
        # fitnesses of individuals in the current generation
        # goal is to be minimized
        self.fitness = []
        for i in range(self.size): self.evaluate_fitness(i)

    # mutation that swaps a random number of pixels (up to amount of pixels in image)
    def mass_swap_mutate(self, ind_pos):
        return np.ctypeslib.as_array(lib.mass_swap(self.population[ind_pos].flatten().ctypes.data_as(c_int_p), int(rd.random() * 1000), self.num_pixels), shape=(self.num_pixels, 3))
    
    # swap mutation that swaps up to double the number of pixels needed to change (fitness)
    # if the fitness is very close to 0, only mutate 1 / 3 pixels
    def smart_swap_mutate(self, ind_pos):
        try:
            max_pixels = self.num_pixels / self.fitness[ind_pos] * 2
        except ZeroDivisionError:
            max_pixels = self.num_pixels / 3
        return np.ctypeslib.as_array(lib.smart_swap(self.population[ind_pos].flatten().ctypes.data_as(c_int_p), int(rd.random() * 1000), int(max_pixels), self.num_pixels), shape=(self.num_pixels, 3))

    """# swap mutation that can only swap up to 10% of the total number of pixels
    def small_swap_mutate(self, ind_pos):
        return np.ctypeslib.as_array(lib.smart_swap(self.population[ind_pos].flatten().ctypes.data_as(c_int_p), int(rd.random() * 1000), int(self.num_pixels * .5), self.num_pixels), shape=(self.num_pixels, 3))
    """
    def save_results(self):
        Image.fromarray(np.reshape(self.best_ind, (self.height, self.width, 3)).astype(np.uint8)).save(f'{self.path}/{self.goal} +- {self.best_fit:.4f}%.png')

    def create_folder_and_save(self, algorithm):
        # create directory for results if one doesnt exist
        self.path = f"{results_folder}/{self.orig_img_file.replace('./images/', '').replace('.png', ' results')}"
        if not os.path.exists(self.path): 
            os.mkdir(self.path)
        self.path = f"{self.path}/{algorithm}"
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.path = f"{self.path}/{self.goal}%"
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.save_results()

class EP_Generation(Generation):
    # stores sorted pos
    sorted_pos = []
    # stores wins of individuals
    wins = []

    # prints wins of each individual
    def print_wins(self):
        for i in range(len(self.fitness)): print(f"Individual {i+1} wins: {self.wins[i]}")
        print()
    
    # prints order of individuals based on wins
    def print_sorted(self):
        for i in range(len(self.fitness)):
            print(f"""Individual: {self.sorted_pos[i] + 1}
    fitness: {self.fitness[self.sorted_pos[i]]}  
    wins: {self.wins[self.sorted_pos[i]]}""")
        print()

    # generates children and adds them to the population
    # also performs repairs on children
    # takes number of iterations to decide which mutations to perform
    def generate_children(self, iters, total_iters):
        if iters < 3 * total_iters / 5:
            for i in range(self.size):
                self.population.append(self.mass_swap_mutate(i) if rd.randint(0,11) <= 8 else self.smart_swap_mutate(i))
                self.evaluate_fitness(self.size)
                self.size += 1
        else:
            for i in range(self.size):
                self.population.append(self.smart_swap_mutate(i) if rd.randint(0,11) <= 8 else self.mass_swap_mutate(i))
                self.evaluate_fitness(self.size)
                self.size += 1

    # round-robin tournament to assign wins to each individual
    # sets q: q = number of opponents to face
    # sets chance: chance of better individual winning (int 0-100)
    def round_robin(self, q, chance):
        self.q = q
        self.wins = []
        #self.wins = lib.round_robin(np.asarray(self.fitness[:self.size]).ctypes.data_as(c_float_p), self.size, int(rd.random()*1000000), q, chance)
        for i in range(self.size): 
            pos_used = []
            self.wins.append(0)
            for r in range(q):
                opp = i
                used = True
                while used:
                    opp = rd.randrange(self.size)
                    if opp is not i:
                        found = False
                        for u in range(r):
                            if pos_used[u] == opp: 
                                found = True
                                u = r
                        used = found
                if self.fitness[i] < self.fitness[opp]: self.wins[i] += 1
                elif self.fitness[i] == self.fitness[opp]:
                    if rd.randrange(2) == 0: self.wins[i] += 1
                pos_used.append(opp)

    # sorts individuals by wins and stores sorted positions
    def sort_wins(self):
        self.sorted_pos = []
        for i in range(self.q, -1, -1):
            for k in range(self.size):
                if self.wins[k] == i: 
                    self.sorted_pos.append(k)

    # survivor selection: takes top N individuals to survive on
    def survivor_select(self):
        self.size = int(self.size / 2)
        self.population = [self.population[self.sorted_pos[i]] for i in range(self.size)]
        self.fitness = [self.fitness[self.sorted_pos[i]] for i in range(self.size)]

# steady state GA
class GA_Generation(Generation):
    # contains individuals selected to be parents
    mating_pool = []
    # contains individuals that will survive
    survivor_pool = []
    
    # PMX crossover
    def pmx_cross(self, p1_pos, p2_pos):
        self.population.append(lib.pmx_cross(self.population[p1_pos].flatten().ctypes.data_as(c_int_p), 
                              self.population[p2_pos].flatten().ctypes.data_as(c_int_p), 
                              self.num_pixels))

    # PMX crossover where the max range of the crossover is up to double 
    # the number of pixels needed to reach the goal percentage from the average of the 
    # parents' fitnesses. if the average fitness is very close to 0, perform order crossover
    def smart_pmx_cross(self, p1_pos, p2_pos):
        try:
            avg_fit = (self.fitness[p1_pos] + self.fitness[p2_pos]) / 2
            max_pixels = self.num_pixels / avg_fit * 2
            self.population.append(lib.smart_pmx_cross(self.population[p1_pos].flatten().ctypes.data_as(c_int_p), 
                              self.population[p2_pos].flatten().ctypes.data_as(c_int_p), 
                              self.num_pixels, int(max_pixels)))
        except ZeroDivisionError:
            self.order_cross(p1_pos, p2_pos)

    def order_cross(self, p1_pos, p2_pos):
        self.population.append(lib.order_cross(self.population[p1_pos].flatten().ctypes.data_as(c_int_p), 
                              self.population[p2_pos].flatten().ctypes.data_as(c_int_p), 
                              self.num_pixels))

    # parent selection: tournament with k opponents  
    # stochastic with fitter individuals have 80% chance of winning
    # once two winners are selected, Smart PMX crossover
    def tournament_select(self, k):
        fit_p = np.asarray(self.fitness[:self.size]).ctypes.data_as(c_float_p)
        for i in range(self.size):
            pos_1 = lib.tournament_select(fit_p, k, self.size)
            pos_2 = lib.tournament_select(fit_p, k, self.size)
            self.population.append(self.smart_pmx_cross(pos_1, pos_2))
        del fit_p

    def mutate_children(self):
        for i in range(self.size):
            self.population[self.size+i] = self.smart_swap_mutate(i)
            self.evaluate_fitness(self.size+i)

    # does not allow duplicates
    def tournament_survive(self, k):
        fit_p = np.asarray(self.fitness[:self.size]).ctypes.data_as(c_float_p)
        survivors = []
        for i in range(self.size): 
            survivor = lib.tournament_select(fit_p, k, self.size)
            while survivor in survivors: survivor = lib.tournament_select(fit_p, k, self.size)
            survivors.append(survivor)
        self.population = [self.population[survivors[i]] for i in range(self.size)]
        self.fitness = [self.fitness[survivors[i]] for i in range(self.size)]
        del fit_p
        del survivors

        # round-robin tournament to assign wins to each individual
    
    # sets q: q = number of opponents to face
    # sets chance: chance of better individual winning (int 0-100)
    def round_robin(self, q, chance):
        self.q = q
        self.wins = []
        #self.wins = lib.round_robin(np.asarray(self.fitness[:self.size]).ctypes.data_as(c_float_p), self.size, int(rd.random()*1000000), q, chance)
        for i in range(self.size*2): 
            pos_used = []
            self.wins.append(0)
            for r in range(q):
                opp = i
                used = True
                while used:
                    opp = rd.randrange(self.size)
                    if opp is not i:
                        found = False
                        for u in range(r):
                            if pos_used[u] == opp: 
                                found = True
                                u = r
                        used = found
                if self.fitness[i] < self.fitness[opp]: self.wins[i] += 1
                elif self.fitness[i] == self.fitness[opp]:
                    if rd.randrange(2) == 0: self.wins[i] += 1
                pos_used.append(opp)

    # sorts individuals by wins and stores sorted positions using lib
    def sort_wins(self):
        #self.sorted_pos = lib.sort_wins(np.asarray(self.wins).ctypes.data_as(c_int_p), len(self.wins), self.q)
        self.sorted_pos = []
        for i in range(self.q, -1, -1):
            for k in range(self.size*2):
                if self.wins[k] == i: 
                    self.sorted_pos.append(k)

    # survivor selection: takes top N individuals to survive on
    def survivor_select(self):
        self.population = [self.population[self.sorted_pos[i]] for i in range(self.size)]
        self.fitness = [self.fitness[self.sorted_pos[i]] for i in range(self.size)]
  
class Greedy_Solution():
    # path to results folder
    path = ' '

    # constructor takes original image's file, the number of individuals to be generated, 
    # the percentage of likeness desired, and the margin of error for 
    # accepting individuals as solutions
    def __init__(self, img_file, goal):
        # original image's file
        self.orig_img_file = img_file
        # original image
        self.orig_image = Image.open(img_file).convert('RGB')
        # width and height of all images
        self.width, self.height = self.orig_image.size
        # number of pixels in all image
        self.num_pixels = self.width * self.height
        # original image's pixels stored as a 1D array (2D if considering each pixel's individual values)
        self.orig_pixels = np.asarray(self.orig_image).reshape((self.num_pixels, 3)).astype(c_uint)
        # percentage of likeness desired
        self.goal = goal

    # generates greedy solution. goes to each position and the chance of a pixel being left
    # alone (not moved) is the goal %. If the position is to be changed it is swapped with a
    # pixel in from an unused location 
    def greedy_generate(self):
        self.solution = np.ctypeslib.as_array(lib.greedy_generate(self.orig_pixels.flatten().ctypes.data_as(c_int_p), c_double(self.goal), int(rd.random() * 1000), self.num_pixels), shape=(self.num_pixels, 3))
    
    # evaluates the fitness of an individual in the population using lib
    # fitness is how close the percentage of likeness an image (to the original)
    # is to the goal percentage of likeness
    def evaluate_fitness(self):
        self.fit = lib.evaluate_fitness(self.solution.flatten().ctypes.data_as(c_int_p), self.orig_pixels.flatten().ctypes.data_as(c_int_p), c_double(self.goal), self.num_pixels)

    def print_solution(self):
        lib.print_individual(self.solution.flatten().ctypes.data_as(c_int_p), self.num_pixels)

    def display_result(self):
        Image.fromarray(np.reshape(self.solution, (self.height, self.width, 3)).astype(np.uint8)).show()

    def save_result(self):
        Image.fromarray(np.reshape(self.solution, (self.height, self.width, 3)).astype(np.uint8)).save(f'{self.path}/{self.goal} +- {self.fit:.4f}%.png')

    def create_folder_and_save(self, algorithm):
        # create directory for results if one doesnt exist
        self.path = f"{results_folder}/{self.orig_img_file.replace('./images/', '').replace('.png', ' results')}"
        if not os.path.exists(self.path): 
            os.mkdir(self.path)
        self.path = f"{self.path}/{algorithm}"
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.path = f"{self.path}/{self.goal}%"
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.save_result()
