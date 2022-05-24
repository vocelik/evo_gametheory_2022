import axelrod as axl
import numpy as np
import re
from scipy.stats import truncnorm

np.random.seed(1)  # important because it determines how the players are shuffled

regex = re.compile("[^a-zA-Z]")  # used for cleaning result output

# MORAN PROCESS SETUP
MAX_ROUNDS = 1000
SEEDS = [
    894,
    201,
    450,
    898,
    588,
    80,
    824,
    218,
    780,
    385,
    559,
    743,
    411,
    536,
    895,
    367,
    461,
    130,
    711,
    612,
    565,
    990,
    897,
    179,
    123,
]
TURNS = 100
MUTATION_RATE = 0.1
NOISE = 0.1
NUMPY_RANDOM_SEEDS = [
    981,
    409,
    953,
    550,
    634,
    598,
    763,
    828,
    638,
    553,
    223,
    332,
    759,
    35,
    579,
    642,
    625,
    123,
    748,
    510,
    85,
    721,
    136,
    537,
    445,
]

# STRATEGY SETUP
STRATEGIES = [axl.Defector, axl.TitForTat, axl.GTFT, axl.Cooperator]
NUMBER_OF_PLAYERS = 10
PLAYERS = [player() for player in range(NUMBER_OF_PLAYERS) for player in STRATEGIES]
np.random.shuffle(PLAYERS)  # randomize list of players

# mass
distributions_mass = {
    "normal": [],
    "pareto": [],
    "uniform": [],
    "symetric_bimodal": [],
    "asymetric_bimodal": [],
    "homo": [],
}

# independence
distributions_independence = {
    "normal": [],
    "pareto": [],
    "uniform": [],
    "symetric_bimodal": [],
    "asymetric_bimodal": [],
    "homo": [],
}

for seed in NUMPY_RANDOM_SEEDS:

    np.random.seed(seed)

    # DISTRIBUTIONS
    ## Pareto
    pareto_upper_bound_mass, pareto_upper_bound_independence = 100, 100
    pareto_population_mass = (np.random.pareto(3.23, pow(10, 6)) + 1) * 0.69
    pareto_population_independence = (np.random.pareto(3.23, pow(10, 6)) + 1) * 0.69

    ## Normal
    normal_mass_lower_bound, normal_mass_upper_bound = 0.01, 1.99
    normal_population_mass = np.random.normal(1, 0.5, pow(10, 6))
    normal_independence_lower_bound, normal_independence_upper_bound = 0.01, 1.99
    normal_population_independence = np.random.normal(1, 0.5, pow(10, 6))

    ## Bimodal
    symetric_bimodal, symetric_prob = [0.5, 1.5], [0.5, 0.5]
    asymetric_bimodal, asymetric_prob = [0.75, 2], [0.8, 0.2]

    ## Uniform
    uniform_population_mass = np.random.uniform(0.13, 1.87, pow(10, 6))
    uniform_population_independence = np.random.normal(0.13, 1.87, pow(10, 6))

    ## Homogenous
    mass_base = 1  # value of mass when distribution is homogenous
    independence_base = 1  # value of independence when distribution is homogenous

    # SIMULATION VALUES
    # mass
    distributions_mass["normal"].append(
        normal_population_mass[
            (normal_population_mass < normal_mass_upper_bound)
            & (normal_population_mass > normal_mass_lower_bound)
        ][: len(PLAYERS)].round(2)
    )
    distributions_mass["pareto"].append(
        pareto_population_mass[pareto_population_mass < pareto_upper_bound_mass][
            : len(PLAYERS)
        ].round(2)
    )
    distributions_mass["uniform"].append(
        uniform_population_mass[: len(PLAYERS)].round(2)
    )
    distributions_mass["symetric_bimodal"].append(
        list(np.random.choice(symetric_bimodal, len(PLAYERS), p=symetric_prob))
    )
    distributions_mass["asymetric_bimodal"].append(
        list(np.random.choice(asymetric_bimodal, len(PLAYERS), p=asymetric_prob))
    )
    distributions_mass["homo"].append([mass_base for _ in range(len(PLAYERS))])

    # independence
    distributions_independence["normal"].append(
        normal_population_mass[
            (normal_population_independence < normal_independence_upper_bound)
            & (normal_population_independence > normal_independence_lower_bound)
        ][: len(PLAYERS)].round(2)
    )
    distributions_independence["pareto"].append(
        pareto_population_independence[
            pareto_population_independence < pareto_upper_bound_independence
        ][: len(PLAYERS)].round(2)
    )
    distributions_independence["uniform"].append(
        uniform_population_independence[: len(PLAYERS)].round(2)
    )
    distributions_independence["symetric_bimodal"].append(
        list(np.random.choice(symetric_bimodal, len(PLAYERS), p=symetric_prob))
    )
    distributions_independence["asymetric_bimodal"].append(
        list(np.random.choice(asymetric_bimodal, len(PLAYERS), p=asymetric_prob))
    )
    distributions_independence["homo"].append(
        [independence_base for _ in range(len(PLAYERS))]
    )
