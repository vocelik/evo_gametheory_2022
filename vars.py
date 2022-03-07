import axelrod as axl
import numpy as np
import re
import random
from scipy.stats import truncnorm

np.random.seed(1) # important because we are generating the mass distributions through numpy


regex = re.compile('[^a-zA-Z]') # used for cleaning result output


# MORAN PROCESS SETUP
MAX_ROUNDS = 5
SEEDS = [1,2,3,4,5]
TURNS = 100
MUTATION_RATE = .1
NOISE = .1

# STRATEGY SETUP
STRATEGIES = [axl.Defector, axl.WinStayLoseShift, axl.TitForTat, axl.GTFT, axl.Cooperator]
NUMBER_OF_PLAYERS = 10
PLAYERS = [player() for player in range(NUMBER_OF_PLAYERS) for player in STRATEGIES]
random.Random(1).shuffle(PLAYERS) # randomize list of players

# DISTRIBUTIONS

## Pareto
pareto_upper_bound_mass, pareto_upper_bound_independence = 2, 2
pareto_population_mass = (np.random.pareto(1, pow(10,6)) + 1) * 1/10
pareto_population_independence = (np.random.pareto(1, pow(10,6)) + 1) * 1/10

## Normal
normal_mass_lower_bound, normal_mass_upper_bound = 1/10, 2
normal_population_mass = np.random.normal(1, 1/3, pow(10,6))
normal_independence_lower_bound, normal_independence_upper_bound = .1, 2
normal_population_independence = np.random.normal(1, 1/3, pow(10,6))

## Bimodal
symetric_bimodal, symetric_prob = [.7, 1.3], [.5, .5]
asymetric_bimodal, asymetric_prob = [.85, 1.6], [.8, .2]

## Uniform
uniform_population_mass = np.random.uniform(.1, 2, pow(10,6))
uniform_population_independence = np.random.normal(.1, 2, pow(10,6))

## Homogenous

mass_base = 1 # value of mass when distribution is homogenous
independence_base = 1 # value of independence when distribution is homogenous

# SIMULATION VALUES

# mass 
distributions_mass = {
    "normal": normal_population_mass[(normal_population_mass < normal_mass_upper_bound) & (normal_population_mass > normal_mass_lower_bound)][:len(PLAYERS)].round(2),
    "pareto": pareto_population_mass[pareto_population_mass < pareto_upper_bound_mass][:len(PLAYERS)].round(2),
    "uniform": uniform_population_mass[:len(PLAYERS)].round(2),
    "symetric_bimodal": list(np.random.choice(symetric_bimodal, len(PLAYERS), p = symetric_prob)),
    "asymetric_bimodal": list(np.random.choice(asymetric_bimodal, len(PLAYERS), p = asymetric_prob)),
    "homo":[mass_base for _ in range(len(PLAYERS))]
}

# independence 
distributions_independence = {
    "normal": normal_population_independence[(normal_population_independence < normal_independence_upper_bound) & (normal_population_independence > normal_independence_lower_bound)][:len(PLAYERS)].round(2),
    "pareto": pareto_population_independence[pareto_population_independence < pareto_upper_bound_independence][:len(PLAYERS)].round(2),
    "uniform": uniform_population_independence[:len(PLAYERS)].round(2),
    "symetric_bimodal": list(np.random.choice(symetric_bimodal, len(PLAYERS), p = symetric_prob)),
    "asymetric_bimodal": list(np.random.choice(asymetric_bimodal, len(PLAYERS), p = asymetric_prob)),
    "homo":[independence_base for _ in range(len(PLAYERS))]
}
