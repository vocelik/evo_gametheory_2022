import axelrod as axl
import numpy as np
import re
from scipy.stats import truncnorm

np.random.seed(2048) # important because it determines how the players are shuffled

regex = re.compile('[^a-zA-Z]') # used for cleaning result output

# MORAN PROCESS SETUP
MAX_ROUNDS = 1000
SEEDS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
TURNS = 100
MUTATION_RATE = .1
NOISE = .1
NUMPY_RANDOM_SEEDS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# STRATEGY SETUP
STRATEGIES = [axl.Defector, axl.TitForTat, axl.GTFT, axl.Cooperator]
NUMBER_OF_PLAYERS = 10
PLAYERS = [player() for player in range(NUMBER_OF_PLAYERS) for player in STRATEGIES]
np.random.shuffle(PLAYERS) # randomize list of players

# mass 
distributions_mass = {
    "normal": [],
    "pareto": [],
    "uniform": [],
    "symetric_bimodal": [],
    "asymetric_bimodal": [],
    "homo": []
}

# independence 
distributions_independence = {
    "normal": [],
    "pareto": [],
    "uniform": [],
    "symetric_bimodal": [],
    "asymetric_bimodal": [],
    "homo": []
}

for seed in NUMPY_RANDOM_SEEDS:

    np.random.seed(seed)

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
    distributions_mass["normal"].append(normal_population_mass[(normal_population_mass < normal_mass_upper_bound) & (normal_population_mass > normal_mass_lower_bound)][:len(PLAYERS)].round(2)) 
    distributions_mass["pareto"].append(pareto_population_mass[pareto_population_mass < pareto_upper_bound_mass][:len(PLAYERS)].round(2))
    distributions_mass["uniform"].append(uniform_population_mass[:len(PLAYERS)].round(2))
    distributions_mass["symetric_bimodal"].append(list(np.random.choice(symetric_bimodal, len(PLAYERS), p = symetric_prob)))
    distributions_mass["asymetric_bimodal"].append(list(np.random.choice(asymetric_bimodal, len(PLAYERS), p = asymetric_prob)))
    distributions_mass["homo"].append([mass_base for _ in range(len(PLAYERS))])

    # independence 
    distributions_independence["normal"].append(normal_population_mass[(normal_population_independence < normal_independence_upper_bound) & (normal_population_independence > normal_independence_lower_bound)][:len(PLAYERS)].round(2)) 
    distributions_independence["pareto"].append(pareto_population_independence[pareto_population_independence < pareto_upper_bound_independence][:len(PLAYERS)].round(2))
    distributions_independence["uniform"].append(uniform_population_independence[:len(PLAYERS)].round(2))
    distributions_independence["symetric_bimodal"].append(list(np.random.choice(symetric_bimodal, len(PLAYERS), p = symetric_prob)))
    distributions_independence["asymetric_bimodal"].append(list(np.random.choice(asymetric_bimodal, len(PLAYERS), p = asymetric_prob)))
    distributions_independence["homo"].append([independence_base for _ in range(len(PLAYERS))])
