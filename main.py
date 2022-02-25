import sys
import time

import axelrod as axl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from collections import Counter
from math import comb

from vars import PLAYERS, SEEDS, MAX_ROUNDS, MUTATION_RATE, NOISE, distributions_mass, distributions_weight, regex, strategies

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

    if sys.argv[2] not in list(distributions_weight.keys()):
        print(f"The population distribution '{sys.argv[2]}' does not exist.")
        print("Either create the distribution in the vars.py file, or choose from the following:")
        for population in list(distributions_weight.keys()):
            print(population)
        
        return False

    set_PLAYER_heterogeneity(PLAYERS, distributions_mass[sys.argv[1]], distributions_weight[sys.argv[2]]) 
    print(f"Successfully created a {sys.argv[1]} population with {len(PLAYERS)} players and a {sys.argv[2]} weight distribution.")

    # save the mass and weight plot
    plt.hist(distributions_mass[sys.argv[1]])
    plt.savefig("results/figures/" + "mass_" + str(sys.argv[1]) + "_weight_" + str(sys.argv[2]) + "_mass_distribution.png")
    plt.clf()
    print("Mass distribution histogram saved.")

    plt.hist(distributions_mass[sys.argv[2]])
    plt.savefig("results/figures/" + "mass_" + str(sys.argv[1]) + "_weight_" + str(sys.argv[2]) + "_weight_distribution.png")
    print("Weight distribution histogram saved.")

    start_time = time.time()
    print_simulation_record()

    # loop over SEEDS and run simulation
    for SEED in SEEDS:
        print(f"Running seed {SEED}...")
        mp = MassBasedMoranProcess(PLAYERS, match_class=MassBaseMatch, turns=200, seed=SEED, mutation_rate=MUTATION_RATE, noise=NOISE)
        
        for i, _ in enumerate(mp):
            if len(mp.population_distribution()) == 1 or i == MAX_ROUNDS - 1:
                break 
                    # save population distribution
        pd.DataFrame(mp.populations).fillna(0).to_csv("results/population_evolution/" + "mass_" + str(sys.argv[1]) + "_weight_" + str(sys.argv[2]) + "_SEED_ " + str(SEED) + "_population_distribution.csv")

        # save outcomes of each round
        df_outcomes = pd.DataFrame(outcomes).fillna(0).rename(columns = {"CC":"coop","CD":"exploit","DC":"exploit_","DD":"defect",})
        df_outcomes['round'] = np.repeat([i + 1 for i in range(MAX_ROUNDS)], comb(len(PLAYERS),2))
        df_outcomes = df_outcomes.groupby(['round']).sum()
        df_outcomes.to_csv("results/outcomes_per_round/" + "mass_" + str(sys.argv[1]) + "_weight_" + str(sys.argv[2]) + "_SEED_ " + str(SEED) + "_outcomes.csv")
        outcomes = []
    
    print(f"Program ran for {round(time.time() - start_time)} seconds.")


def set_PLAYER_heterogeneity(PLAYERS, masses, weights, ids = [i for i in range(len(PLAYERS))]):

    for PLAYER, id, mass, weight in zip(PLAYERS, ids, masses, weights):
        setattr(PLAYER, "id", id + 1)
        setattr(PLAYER, "mass", mass)
        setattr(PLAYER, "weight", weight)


def print_simulation_record():
    print("-" * 75)
    print("\tStarting simulations with the following parameters:")
    print(f"\tmax rounds:{MAX_ROUNDS}")
    print(f"\tseed:{[seed for seed in SEEDS]}")
    print(f"\tmutation rate: {MUTATION_RATE}")
    print(f"\tnoise : {NOISE}")
    print(f"\tmass: {sys.argv[1]} distribution")
    print(f"\tweight: {sys.argv[2]} distribution")
    print(f"\tnumber of players: {len(PLAYERS)}")
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
         final_scores = [score + (PLAYER.mass * PLAYER.weight) for PLAYER, score in zip(self.players, mass_scores)]
         return final_scores


class MassBasedMoranProcess(axl.MoranProcess):
     """Axelrod MoranProcess class """
     def __next__(self):
         set_PLAYER_heterogeneity(self.players, distributions_mass[sys.argv[1]], distributions_weight[sys.argv[2]])
         super().__next__()
         return self


if __name__ == "__main__":
    main()