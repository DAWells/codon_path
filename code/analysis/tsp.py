# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 20:07:15 2021

mlrose travelling salesman problem

@author: David.Wells
"""

import mlrose_hiive as mlrose
import numpy as np

# City co-ordinates
coords_list = [(1,1), (2,5), (9,5),(4,5), (8,2)]

# Initialise fitness function using co-ordinates
fitness_coords = mlrose.TravellingSales(coords = coords_list)

# Define optimisation object
problem_fit = mlrose.TSPOpt(length=5, fitness_fn=fitness_coords, maximize=False)

# Solve the problem
best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem_fit, random_state=2)
