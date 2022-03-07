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
        print("Please specify what population the agents should be drawn from and what independence should be given to PLAYERS.")
        return False

    if sys.argv[1] not in list(distributions_mass.keys()):
        print(f"The population distribution '{sys.argv[1]}' does not exist.")
        print("Either create the distribution in the vars.py file, or choose from the following:")
        for population in list(distributions_mass.keys()):
            print(population)
        return False

    if sys.argv[2] not in list(distributions_independence.keys()):
        print(f"The population distribution '{sys.argv[2]}' does not exist.")
        print("Either create the distribution in the vars.py file, or choose from the following:")
        for population in list(distributions_independence.keys()):
            print(population)
        return False

    # set player heterogeneity mass and independence and save the players
    set_PLAYER_heterogeneity(PLAYERS, distributions_mass[sys.argv[1]], distributions_independence[sys.argv[2]]) 
    save_population_setup()

    # save the mass and independence plots
    save_initialized_plot()

    # the simulation record
    print_simulation_record()

    # start time and keep track of how long the simulations run
    start_time = time.time()

    # loop over seeds and run simulation
    for SEED in SEEDS:
        print(f"Running seed {SEED}...")
        mp = massBasedMoranProcess(PLAYERS, match_class=massBasedMatch, turns=TURNS, seed=SEED, mutation_rate=MUTATION_RATE, noise=NOISE)
        
        # loop over moran process until a single strategy dominates the population or max round is reached
        for i, _ in enumerate(mp):
            if len(mp.population_distribution()) == 1 or i == MAX_ROUNDS - 1:
                break 

        rounds_played = i
        # save population distribution
        pd.DataFrame(mp.populations).fillna(0).astype(int).to_csv("results/population_evolution/seed_ " + str(SEED) + "_mass_" + str(sys.argv[1]) + "_independence_" + str(sys.argv[2]) + "_population_distribution.csv")

        # save outcomes of each round
        df_outcomes = pd.DataFrame(outcomes).fillna(0).rename(columns = {"CC":"coop","CD":"exploit","DC":"exploit_","DD":"defect",})
        df_outcomes['round'] = np.repeat([i + 1 for i in range(rounds_played + 1)], comb(len(PLAYERS),2))
        df_outcomes['seed'] = SEED
        df_outcomes = df_outcomes.groupby(['round','seed']).sum()
        df_outcomes = df_outcomes.astype(int)
        df_outcomes.to_csv("results/outcomes_per_round/seed_" + str(SEED) + "_mass_"  + str(sys.argv[1]) + "_independence_" + str(sys.argv[2]) + "_outcomes.csv")
        outcomes = []
    
    # show how long simulations took
    print(f"Program ran for {round((time.time() - start_time) / 3600,2)} hours.")


def set_PLAYER_heterogeneity(PLAYERS, masses, independences, ids = [i for i in range(len(PLAYERS))]):
    """
    This functions creates a heterogenous population by adding mass and independence to the player object.
    The object characteristics are used to calculate final scores in the massBaseMatch object.
    """

    for PLAYER, id, mass, independence in zip(PLAYERS, ids, masses, independences):
        setattr(PLAYER, "id", id + 1)
        setattr(PLAYER, "mass", mass)
        setattr(PLAYER, "independence", independence)


def save_initialized_plot():
    # save the mass and independence plot
    plt.hist(distributions_mass[sys.argv[1]])
    plt.savefig("results/figures/mass/" + str(sys.argv[1]) + "_mass_distribution.png")
    plt.clf()
    plt.hist(distributions_mass[sys.argv[2]])
    plt.savefig("results/figures/independence/" + str(sys.argv[2]) + "_independence_distribution.png")
    print("Mass and independence histograms saved.")


def save_population_setup():
    data = {
        "player_id": [player.id for player in PLAYERS],
        "player_strategy": [player for player in PLAYERS],
        "mass": [player.mass for player in PLAYERS],
        "independence": [player.independence for player in PLAYERS],
        "ratio": [round(player.mass * player.independence,2) for player in PLAYERS]
    }

    df = pd.DataFrame(data=data)
    df.to_csv("results/population_setup/" + "mass_" + str(sys.argv[1]) + "_independence_" + str(sys.argv[2]) + "_POPULATION_SETUP.csv")
    print("Population setup saved.")


def print_simulation_record():
    print("-" * 75)
    print("\tStarting simulations with the following parameters:")
    print(f"\tMax rounds: {MAX_ROUNDS}")
    print(f"\tTurns: {TURNS}")
    print(f"\tSeeds: {[seed for seed in SEEDS]}")
    print(f"\tMutation rate: {MUTATION_RATE}")
    print(f"\tNoise: {NOISE}")
    print(f"\tmass: {sys.argv[1]} distribution")
    print(f"\tindependence: {sys.argv[2]} distribution")
    print(f"\tNumber of players: {len(PLAYERS)}")
    print(f"\tStrategies:")
    for strategy in STRATEGIES:
        print(f"\t\t{strategy()}")
    print("-" * 75)


class massBasedMatch(axl.Match):
     """Axelrod Match object with a modified final score function to enable mass to influence the final score as a multiplier"""
     def final_score_per_turn(self): 
         outcomes.append(Counter([regex.sub('',str(i)) for i in self.result]))    
         base_scores = axl.Match.final_score_per_turn(self)
         mass_scores = [PLAYER.mass * score for PLAYER, score in zip(self.players[::-1], base_scores)] # list reversed so opponent profits from mass
         return [score + (PLAYER.mass * PLAYER.independence) for PLAYER, score in zip(self.players, mass_scores)] # list not reversed so player profits from his mass * independence


class massBasedMoranProcess(axl.MoranProcess):
     """Axelrod MoranProcess class """
     def __next__(self):
         set_PLAYER_heterogeneity(self.players, distributions_mass[sys.argv[1]], distributions_independence[sys.argv[2]])
         super().__next__()
         return self


if __name__ == "__main__":
    main()