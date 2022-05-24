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
count_population = 0


def main():

    global outcomes
    global count_population

    # check if the script was called correctly
    if len(sys.argv) < 5:
        print("You have not inputted all necessary terminal arguments.")
        return False

    if sys.argv[1] not in list(distributions_mass.keys()):
        print(f"The population distribution '{sys.argv[1]}' does not exist.")
        print(
            "Either create the distribution in the vars.py file, or choose from the following:"
        )
        for population in list(distributions_mass.keys()):
            print(population)
        return False

    if sys.argv[2] not in list(distributions_independence.keys()):
        print(f"The population distribution '{sys.argv[2]}' does not exist.")
        print(
            "Either create the distribution in the vars.py file, or choose from the following:"
        )
        for population in list(distributions_independence.keys()):
            print(population)
        return False

    if len(NUMPY_RANDOM_SEEDS) != len(SEEDS):
        print(
            "The length of population seeds must be equal to the length of tournament seeds."
        )
        return False

    start_time = time.time()

    for i, numpy_seed in enumerate(NUMPY_RANDOM_SEEDS):

        # set player heterogeneity mass and independence and save the players
        set_PLAYER_heterogeneity(
            PLAYERS,
            distributions_mass[sys.argv[1]][count_population],
            distributions_independence[sys.argv[2]][count_population],
        )
        save_population_setup()

        # save the mass and independence plots
        save_initialized_plot()

        # the simulation record
        if count_population == 0:
            print_simulation_record()

        # loop over seeds and run simulation
        for _ in range(len(SEEDS)):
            SEED = SEEDS[i]
            print(f"Running seed {SEED}...")
            mp = massBasedMoranProcess(
                PLAYERS,
                match_class=massBasedMatch,
                turns=TURNS,
                seed=SEED,
                mutation_rate=MUTATION_RATE,
                noise=NOISE,
            )

            # loop over moran process until a single strategy dominates the population or max round is reached
            for i, _ in enumerate(mp):
                if len(mp.population_distribution()) == 1 or i == MAX_ROUNDS - 1:
                    break

            rounds_played = i
            # save population distribution
            pd.DataFrame(mp.populations).fillna(0).astype(int).to_csv(
                "results/population_evolution/seed_ "
                + str(SEED)
                + "_mass_"
                + str(sys.argv[1])
                + "_independence_"
                + str(sys.argv[2])
                + "_mass_weight_"
                + str(sys.argv[3])
                + "_independence_weight_"
                + str(sys.argv[4])
                + "_population_seed_"
                + str(numpy_seed)
                + ".csv"
            )

            # save outcomes of each round
            df_outcomes = (
                pd.DataFrame(outcomes)
                .fillna(0)
                .rename(
                    columns={
                        "CC": "coop",
                        "CD": "exploit",
                        "DC": "exploit_",
                        "DD": "defect",
                    }
                )
            )
            df_outcomes["round"] = np.repeat(
                [i + 1 for i in range(rounds_played + 1)], comb(len(PLAYERS), 2)
            )
            df_outcomes["seed"] = SEED
            df_outcomes = df_outcomes.groupby(["round", "seed"]).sum()
            df_outcomes = df_outcomes.astype(int)
            df_outcomes.to_csv(
                "results/outcomes_per_round/seed_"
                + str(SEED)
                + "_mass_"
                + str(sys.argv[1])
                + "_independence_"
                + str(sys.argv[2])
                + "_mass_weight_"
                + str(sys.argv[3])
                + "_independence_weight_"
                + str(sys.argv[4])
                + "_outcomes_"
                + "population_seed_"
                + str(numpy_seed)
                + ".csv"
            )
            outcomes = []
            break

        count_population += 1

    # show how long simulations took
    print(f"Program ran for {round((time.time() - start_time) / 3600,2)} hours.")


def set_PLAYER_heterogeneity(
    PLAYERS, masses, independences, ids=[i for i in range(len(PLAYERS))]
):
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
    plt.hist(distributions_mass[sys.argv[1]][count_population])
    plt.savefig(
        "results/figures/mass/"
        + str(sys.argv[1])
        + "_mass_distribution_"
        + "population_seed_"
        + str(NUMPY_RANDOM_SEEDS[count_population])
        + ".png"
    )
    plt.clf()
    plt.hist(distributions_independence[sys.argv[2]][count_population])
    plt.savefig(
        "results/figures/independence/"
        + str(sys.argv[2])
        + "_independence_distribution_"
        + "population_seed_"
        + str(NUMPY_RANDOM_SEEDS[count_population])
        + ".png"
    )
    plt.clf()
    print(
        f"Mass and independence histograms saved from seed: {NUMPY_RANDOM_SEEDS[count_population]}."
    )


def save_population_setup():
    data = {
        "player_id": [player.id for player in PLAYERS],
        "player_strategy": [player for player in PLAYERS],
        "mass": [player.mass for player in PLAYERS],
        "independence": [player.independence for player in PLAYERS],
        "ratio": [round(player.mass * player.independence, 2) for player in PLAYERS],
    }

    df = pd.DataFrame(data=data)
    df.to_csv(
        "results/population_setup/"
        + "mass_"
        + str(sys.argv[1])
        + "_independence_"
        + str(sys.argv[2])
        + "_POPULATION_SETUP_"
        + "population_seed_"
        + str(NUMPY_RANDOM_SEEDS[count_population])
        + ".csv"
    )
    print("Population setup saved.")


def print_simulation_record():
    print("-" * 75)
    print("\tStarting simulations with the following parameters:")
    print(f"\tMax rounds: {MAX_ROUNDS}")
    print(f"\tTurns: {TURNS}")
    print(f"\tSeeds: {[seed for seed in SEEDS]}")
    print(f"\tPopulations: {len(NUMPY_RANDOM_SEEDS)}")
    print(f"\tMutation rate: {MUTATION_RATE}")
    print(f"\tNoise: {NOISE}")
    print(f"\tmass: {sys.argv[1]} distribution")
    print(f"\tindependence: {sys.argv[2]} distribution")
    print(f"\tmass weight: {sys.argv[3]}")
    print(f"\tindependence weight: {sys.argv[4]}")
    print(f"\tNumber of players: {len(PLAYERS)}")
    print(f"\tStrategies:")
    for strategy in STRATEGIES:
        print(f"\t\t{strategy()}")
    print("-" * 75)


class massBasedMatch(axl.Match):
    """Axelrod Match object with a modified final score function to enable mass to influence the final score as a multiplier"""

    def final_score_per_turn(self):
        outcomes.append(Counter([regex.sub("", str(i)) for i in self.result]))
        base_scores = axl.Match.final_score_per_turn(self)
        mass_scores = [
            PLAYER.mass * score * float(sys.argv[3])
            for PLAYER, score in zip(self.players[::-1], base_scores)
        ]  # list reversed so opponent profits from mass
        return [
            score + (PLAYER.mass * PLAYER.independence * float(sys.argv[4]))
            for PLAYER, score in zip(self.players, mass_scores)
        ]  # list not reversed so player profits from his mass * independence


class massBasedMoranProcess(axl.MoranProcess):
    """Axelrod MoranProcess class"""

    def __next__(self):
        set_PLAYER_heterogeneity(
            self.players,
            distributions_mass[sys.argv[1]][count_population],
            distributions_independence[sys.argv[2]][count_population],
        )
        super().__next__()
        return self


if __name__ == "__main__":
    main()
