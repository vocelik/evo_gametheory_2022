import axelrod as axl
import numpy as np
import re
import random

from scipy.stats import truncnorm

np.random.seed(1) # important because we are generating the mass distributions through numpy

regex = re.compile('[^a-zA-Z]')
strategies = [axl.Defector, axl.TitForTat, axl.Cooperator, axl.WinStayLoseShift, axl.Adaptive, axl.Grudger, axl.ZDExtortion, axl.ZDGen2, axl.GTFT, axl.Bully]
MAX_ROUNDS = 1000
SEEDS = [1,2,3,4,5,6,7,8,9,10]
TURNS = 200
MUTATION_RATE = .1
NOISE = .1
MASS_BASE = 1
WEIGHT_BASE = 1
NUMBER_OF_PLAYERS = 5
PLAYERS = [player() for player in range(NUMBER_OF_PLAYERS) for player in strategies]
random.Random(1).shuffle(PLAYERS)


def get_truncated_normal(mean, sd, low, upp):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


pareto_distribution = (np.random.pareto(1, 1000000) + 1) * .1
pareto_sample = [round(i,1) for i in pareto_distribution]
pareto_sample_truncated = [i for i in pareto_sample if i <= 2]


M_trunc = get_truncated_normal(mean=1, sd = 1, low = 0.1, upp = 2)
W_trunc = get_truncated_normal(mean=1, sd = 1, low = 0.1, upp = 2)


distributions_mass = {
    "normal": [round(np.random.choice([i for i in M_trunc.rvs(1000)]), 1) for _ in range(len(PLAYERS))],
    "pareto": [round(np.random.choice(pareto_sample_truncated), 1) for _ in range(len(PLAYERS))],
    "homo":[MASS_BASE for _ in range(len(PLAYERS))]
}


distributions_weight = {
    "normal": [round(np.random.choice([i for i in W_trunc.rvs(1000)]), 1) for _ in range(len(PLAYERS))],
    "pareto": [round(np.random.choice(pareto_sample_truncated), 1) for _ in range(len(PLAYERS))],
    "homo":[MASS_BASE for _ in range(len(PLAYERS))]
}
