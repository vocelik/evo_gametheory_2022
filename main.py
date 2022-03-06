import sys
import time

import axelrod as axl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import Counter
from math import comb
from vars import *

# import note, strategies only have memory inside a match class
outcomes = []

def main():

    global outcomes

    # check if the script was called correctly 
    if len(sys.argv) < 3:
        print("Please specify what population the agents should be drawn from and what weight should be given to PLAYERS.")
        return False

    if sys.argv[1] not in list(distributions_mass.keys()):
        print(f"The population distribution '{sys.argv[1]}' does not exist.")
        print("Either create the distribution in the vars.py file, or choose from the following:")
        for population in list(distributions_mass.keys()):
            print(population)
        return False

    if sys.argv[2] not in list(distributions_weight.keys()):
        print(f"The population distribution '{sys.argv[2]}' does not exist.")
        print("Either create the distribution in the vars.py file, or choose from the following:")
        for population in list(distributions_weight.keys()):
            print(population)
        return False

    # set player heterogeneity mass and weight
    set_PLAYER_heterogeneity(PLAYERS, distributions_mass[sys.argv[1]], distributions_weight[sys.argv[2]]) 

    # save the mass and weight plots
    save_initialized_plot()

    # the simulation record
    print_simulation_record()

    # start time and keep track of how long the simulations run
    start_time = time.time()

    # loop over SEEDS and run simulation
    for SEED in SEEDS:
        print(f"Running seed {SEED}...")
        mp = MassBasedMoranProcess(PLAYERS, match_class=MassBaseMatch, turns=TURNS, seed=SEED, mutation_rate=MUTATION_RATE, noise=NOISE)
        
        # loop over moran process until a single strategy dominates the population or max round is reached
        for i, _ in enumerate(mp):
            if len(mp.population_distribution()) == 1 or i == MAX_ROUNDS - 1:
                break 
        # save population distribution
        pd.DataFrame(mp.populations).fillna(0).to_csv("results/population_evolution/" + "mass_" + str(sys.argv[1]) + "_weight_" + str(sys.argv[2]) + "_SEED_ " + str(SEED) + "_population_distribution.csv")

        # save outcomes of each round
        df_outcomes = pd.DataFrame(outcomes).fillna(0).rename(columns = {"CC":"coop","CD":"exploit","DC":"exploit_","DD":"defect",})
        df_outcomes['round'] = np.repeat([i + 1 for i in range(MAX_ROUNDS)], comb(len(PLAYERS),2))
        df_outcomes = df_outcomes.groupby(['round']).sum()
        df_outcomes = df_outcomes.astype(int)
        df_outcomes.to_csv("results/outcomes_per_round/" + "mass_" + str(sys.argv[1]) + "_weight_" + str(sys.argv[2]) + "_SEED_ " + str(SEED) + "_outcomes.csv")
        outcomes = []
    
    # show how long simulations took
    print(f"Program ran for {round(time.time() - start_time) / 3600 } hours.")


def set_PLAYER_heterogeneity(PLAYERS, masses, weights, ids = [i for i in range(len(PLAYERS))]):

    """
    This functions creates a heterogenous population by adding mass and weight to the player object.
    The object characteristics are used to calculate final scores in the MassBaseMatch object.
    """

    for PLAYER, id, mass, weight in zip(PLAYERS, ids, masses, weights):
        setattr(PLAYER, "id", id + 1)
        setattr(PLAYER, "mass", mass)
        setattr(PLAYER, "weight", weight)


def save_initialized_plot():

    # save the mass and weight plot
    plt.hist(distributions_mass[sys.argv[1]])
    plt.savefig("results/figures/" + "mass_" + str(sys.argv[1]) + "_weight_" + str(sys.argv[2]) + "_mass_distribution.png")
    plt.clf()
    print("Mass distribution histogram saved.")

    plt.hist(distributions_mass[sys.argv[2]])
    plt.savefig("results/figures/" + "mass_" + str(sys.argv[1]) + "_weight_" + str(sys.argv[2]) + "_weight_distribution.png")
    print("Weight distribution histogram saved.")


def print_simulation_record():

    print("-" * 75)
    print("\tStarting simulations with the following parameters:")
    print(f"\tMax rounds: {MAX_ROUNDS}")
    print(f"\tTurns: {TURNS}")
    print(f"\tSeeds: {[seed for seed in SEEDS]}")
    print(f"\tMutation rate: {MUTATION_RATE}")
    print(f"\tNoise: {NOISE}")

    print(f"\tMass: {sys.argv[1]} distribution")
    if sys.argv[1] == "homo":
        print(f"\t\tMass base: {distributions_mass[sys.argv[1]][0]}")
    if sys.argv[1] == "pareto":
        print(f"\t\tPareto shape: {PARETO_SHAPE_MASS}")
        print(f"\t\tPareto scale: {PARETO_SCALE_MASS}")

    print(f"\tWeight: {sys.argv[2]} distribution")
    if sys.argv[2] == "homo":
        print(f"\t\tWeight base: {distributions_weight[sys.argv[2]][0]}")
    if sys.argv[2] == "pareto":
        print(f"\t\tPareto shape: {PARETO_SHAPE_WEIGHT}")
        print(f"\t\tPareto scale: {PARETO_SCALE_WEIGHT}")

    print(f"\tNumber of players: {len(PLAYERS)}")
    print(f"\tStrategies:")
    for strategy in strategies:
        print(f"\t\t{strategy()}")
    print("-" * 75)


class MassBaseMatch(axl.Match):
     """Axelrod Match object with a modified final score function to enable mass to influence the final score as a multiplier"""
     def final_score_per_turn(self): 
         outcomes.append(Counter([regex.sub('',str(i)) for i in self.result]))    
         base_scores = axl.Match.final_score_per_turn(self)
         mass_scores = [PLAYER.mass * score for PLAYER, score in zip(self.players[::-1], base_scores)]
         return [score + (PLAYER.mass * PLAYER.weight) for PLAYER, score in zip(self.players, mass_scores)]


class MassBasedMoranProcess(axl.MoranProcess):
     """Axelrod MoranProcess class """
     def __next__(self):
         set_PLAYER_heterogeneity(self.players, distributions_mass[sys.argv[1]], distributions_weight[sys.argv[2]])
         super().__next__()
         return self


if __name__ == "__main__":
    main()