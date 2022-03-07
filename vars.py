import axelrod as axl
import numpy as np
import re
from scipy.stats import truncnorm

np.random.seed(2048) # important because we are generating the mass distributions through numpy


regex = re.compile('[^a-zA-Z]') # used for cleaning result output


# MORAN PROCESS SETUP
MAX_ROUNDS = 1000
SEEDS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
TURNS = 100
MUTATION_RATE = .1
NOISE = .1

# STRATEGY SETUP
STRATEGIES = [axl.Defector, axl.TitForTat, axl.GTFT, axl.Cooperator]
NUMBER_OF_PLAYERS = 10
PLAYERS = [player() for player in range(NUMBER_OF_PLAYERS) for player in STRATEGIES]
np.random.shuffle(PLAYERS) # randomize list of players

# DISTRIBUTIONS

## Pareto
pareto_upper_bound_mass, pareto_upper_bound_independence = 100, 100
pareto_population_mass = (np.random.pareto(3.23, pow(10,6)) + 1) * .69
pareto_population_independence = (np.random.pareto(3.23, pow(10,6)) + 1) * .69

## Normal
normal_mass_lower_bound, normal_mass_upper_bound = .01, 1.99
normal_population_mass = np.random.normal(1, .5, pow(10,6))
normal_independence_lower_bound, normal_independence_upper_bound = .01, 1.99
normal_population_independence = np.random.normal(1, 0.5, pow(10,6))

## Bimodal
symetric_bimodal, symetric_prob = [.5, 1.5], [.5, .5]
asymetric_bimodal, asymetric_prob = [.75, 2], [.8, .2]

## Uniform
uniform_population_mass = np.random.uniform(.13, 1.87, pow(10,6))
uniform_population_independence = np.random.normal(.13, 1.87, pow(10,6))

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
