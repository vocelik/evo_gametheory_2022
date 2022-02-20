"""
This file contains all the required variables utilized in the simulation.
"""
from scipy.stats import truncnorm

import matplotlib.pyplot as plt

import random 
import axelrod as axl
import numpy as np

np.random.seed(1) # important because we are generating the mass distributions through numpy

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

#players
number = 10
defectors = [axl.Defector() for i in range(number)]
cooperators = [axl.Cooperator() for i in range(number)]
tit_for_tatters = [axl.TitForTat() for i in range(number)]
generous_tit_for_tatters = [axl.GTFT() for i in range(number)]
players = defectors + cooperators + tit_for_tatters + generous_tit_for_tatters

random.Random(1).shuffle(players)

pareto_distribution = (np.random.pareto(1, 100000) + 1) * 2.5
pareto_sample = [round(i,1) for i in pareto_distribution]
pareto_sample_truncated = [i for i in pareto_sample if i <= 10]

M_trunc = get_truncated_normal(mean=5, sd = 12.5, low = 0.1, upp = 10)

#masses
distributions = {
    "pareto": [round(np.random.choice(pareto_sample_truncated), 1) for _ in range(len(players))],
    "normal": [round(np.random.choice([i for i in M_trunc.rvs(1000)]), 1) for _ in range(len(players))],
    "homo":[1 for _ in range(len(players))]
}
