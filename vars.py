import axelrod as axl
import numpy as np
import re
import random

from scipy.stats import truncnorm

np.random.seed(1) # important because we are generating the mass distributions through numpy

regex = re.compile('[^a-zA-Z]')
strategies = [axl.Defector, axl.TitForTat, axl.Cooperator, axl.WinStayLoseShift, axl.Adaptive, axl.Grudger, axl.ZDExtortion, axl.ZDGen2, axl.GTFT, axl.Bully]
NUMBER_OF_PLAYERS = 5
MAX_ROUNDS = 1
SEEDS = [1,2,3,4,5,6,7,8,9,10]
TURNS = 200
MUTATION_RATE = .1
NOISE = .1
MASS_BASE = 4
WEIGHT_BASE = 10
PLAYERS = [player() for player in range(NUMBER_OF_PLAYERS) for player in strategies]
random.Random(1).shuffle(PLAYERS)


def get_truncated_normal(mean, sd, low, upp):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


# PARETO SETUP
PARETO_SHAPE_MASS = 1
PARETO_SCALE_MASS = .1
PARETO_UPPER_BOUND_MASS = 2
pareto_distribution_MASS = (np.random.pareto(PARETO_SHAPE_MASS, 1000000) + 1) * PARETO_SCALE_MASS
pareto_sample_MASS = [round(i,1) for i in pareto_distribution_MASS]
pareto_sample_truncated_MASS = [i for i in pareto_sample_MASS if i <= PARETO_UPPER_BOUND_MASS]

PARETO_SHAPE_WEIGHT = 6
PARETO_SCALE_WEIGHT = .5
PARETO_UPPER_BOUND_WEIGHT = 2
pareto_distribution_WEIGHT = (np.random.pareto(PARETO_SHAPE_WEIGHT, 1000000) + 1) * PARETO_SCALE_WEIGHT
pareto_sample_WEIGHT = [round(i,1) for i in pareto_distribution_WEIGHT]
pareto_sample_truncated_WEIGHT = [i for i in pareto_sample_WEIGHT if i <= PARETO_UPPER_BOUND_WEIGHT]


# TRUNCATED NORMAL DISTRIBUTION SETUP
MASS_TRUNCATED_DISTRIBUTION = get_truncated_normal(mean=1, sd = 1, low = 0.1, upp = 2)
WEIGHT_TRUNCATED_DISTRIBUTION = get_truncated_normal(mean=1, sd = 1, low = 0.1, upp = 2)


#BIMODAL DISTRIBUTION
BIMODAL_STRONG_PLAYER_MASS = 2
BIMODAL_WEAK_PLAYER_MASS = .1
BIMODAL_STRONG_PLAYER_WEIGHT = 2
BIMODAL_WEAK_PLAYER_WEIGHT = .1

# DISTRIBUTIONS OF MASS AND WEIGHT
distributions_mass = {
    "normal": [round(np.random.choice([i for i in MASS_TRUNCATED_DISTRIBUTION.rvs(1000)]), 1) for _ in range(len(PLAYERS))],
    "pareto": [round(np.random.choice(pareto_sample_truncated_MASS), 1) for _ in range(len(PLAYERS))],
    "bimodal": [BIMODAL_STRONG_PLAYER_MASS for _ in range( int( len(PLAYERS) / 2) ) ] + [BIMODAL_WEAK_PLAYER_MASS for _ in range( int( len(PLAYERS) / 2) ) ],
    "homo":[MASS_BASE for _ in range(len(PLAYERS))]
}


distributions_weight = {
    "normal": [round(np.random.choice([i for i in WEIGHT_TRUNCATED_DISTRIBUTION.rvs(1000)]), 1) for _ in range(len(PLAYERS))],
    "pareto": [round(np.random.choice(pareto_sample_truncated_WEIGHT), 1) for _ in range(len(PLAYERS))],
    "bimodal": [BIMODAL_STRONG_PLAYER_WEIGHT for _ in range( int( len(PLAYERS) / 2) ) ] + [BIMODAL_WEAK_PLAYER_WEIGHT for _ in range( int( len(PLAYERS) / 2) ) ],
    "homo":[WEIGHT_BASE for _ in range(len(PLAYERS))]
}
