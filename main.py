import sys
import time
import axelrod as axl
import pandas as pd
import matplotlib.pyplot as plt

from vars import players, distributions
from collections import Counter
from typing import Callable, List
from axelrod import Game, Player
from axelrod.deterministic_cache import DeterministicCache
from axelrod.graph import Graph

max_rounds = 1000
seeds = [1,2,3,4,5,6,7,8,9,10]

def main():

    # check if the script was called correctly 
    if len(sys.argv) < 3:
        print("Please specify what population the agents should be drawn from and what weight should be given to players.")
        return False

    if sys.argv[1] not in list(distributions.keys()):
        print(f"The population distribution '{sys.argv[1]}' does not exist.")
        print("Either create the distribution in the vars.py file, or choose from the following:")
        for population in list(distributions.keys()):
            print(population)
        
        return False
    
    # set up the simulation
    population_type = sys.argv[1]
    masses = distributions[population_type]
    weight = sys.argv[2]

    # set the heterogeneity of mass and weight
    set_player_attributes(players, masses, weight)
    print(f"Successfully created a {sys.argv[1]} population with {len(players)} players.")

    # show the distribution of players
    for player in players:
        print(f"Player {player} with id {player.id} has mass {player.mass} and weight {player.weight}.")

    # save the plot
    plt.hist(masses)
    #plt.savefig("results/" + population_type + "/distribution.png")
    print("Population distribution histogram saved.")

    # Begin simulation and track elapsed time
    print("Beginning simulation...")
    start_time = time.time()
    for seed in seeds:

        print(f"Program has run for {round(time.time() - start_time)} seconds ")
        print(f"Now running seed: {seed}...")

        mp = HeterogenousMoranProcess(players, match_class=HeterogenousMatch, turns=200, seed=seed, mutation_rate=.1, noise=.1)
        df_seed = pd.DataFrame()
        for i, _ in enumerate(mp):
            
            df = pd.DataFrame.from_dict(mp.population_distribution(), orient='index').reset_index()
            df = df.rename(columns={'index':'strategy', 0:'count'})
            df['round'] = i
            df = df.pivot(index = 'round', columns = 'strategy', values = 'count')
            df_seed = df_seed.append(df)
            if len(mp.population_distribution()) == 1 or i == max_rounds:
                break    

        df = pd.DataFrame(mp.outcomes_per_round)
        df.to_csv("results/" + population_type + "/outcomes_" + "seed_" + str(seed) + "_weight_" + str(weight) + ".csv")
        df_seed.to_csv("results/" + population_type + "/population_distribution_" + "seed_" + str(seed) + "_weight_" + str(weight) + ".csv")
    print("Simulations completed.")
    print(f"Program has run for {round(time.time() - start_time) / 60} minutes")

    return True


def set_player_attributes(players, masses, weight):

    for i, (player, mass) in enumerate(zip(players, masses)):
        setattr(player, "mass", mass)
        setattr(player, "id", "player " + str(i))
        setattr(player, "weight", weight)


class HeterogenousMatch(axl.Match):
    """Axelrod Match object with a modified final score function to enable mass to influence the final score as a multiplier"""
    def final_score_per_turn(self):
        base_scores = axl.Match.final_score_per_turn(self)
        # here we flip the list because we want the mass of the opponent to be added to the payoff of the player.
        mass_scores = [round(player.mass * score,2) for player, score in zip(self.players[::-1], base_scores)] 
        # here we do not flip the list because we want the weight of the player to be multiplied with his own mass. 
        return [round(score + (player.mass * player.weight),2) for player, score in zip(self.players, mass_scores)]


class HeterogenousMoranProcess(axl.MoranProcess):
    """Axelrod MoranProcess class """
    def __init__(self, 
                players: List[Player], 
                turns: int = ..., 
                prob_end: float = None,
                noise: float = 0, 
                game: Game = None, 
                deterministic_cache: DeterministicCache = None, 
                mutation_rate: float = 0, 
                mode: str = "bd", 
                interaction_graph: Graph = None, 
                reproduction_graph: Graph = None, 
                fitness_transformation: Callable = None, 
                mutation_method="transition", 
                stop_on_fixation=True, 
                seed=None, 
                match_class=...) -> None:
        super().__init__(players, 
                        turns=turns, 
                        prob_end=prob_end, 
                        noise=noise, 
                        game=game, 
                        deterministic_cache=deterministic_cache, 
                        mutation_rate=mutation_rate, 
                        mode=mode, 
                        interaction_graph=interaction_graph, 
                        reproduction_graph=reproduction_graph, 
                        fitness_transformation=fitness_transformation, 
                        mutation_method=mutation_method, 
                        stop_on_fixation=stop_on_fixation,
                        seed=seed, 
                        match_class=match_class)
        self.outcomes_per_round = dict()
        self.round = 0

    def score_all(self) -> List:
            """Plays the next round of the process. Every player is paired up
            against every other player and the total scores are recorded.
            Returns
            -------
            scores:
                List of scores for each player
            """
            N = len(self.players)
            scores = [0] * N
            self.round += 1
            outcome = []
            for i, j in self._matchup_indices():
                player1 = self.players[i]
                player2 = self.players[j]
                match = self.match_class(
                    (player1, player2),
                    turns=self.turns,
                    prob_end=self.prob_end,
                    noise=self.noise,
                    game=self.game,
                    deterministic_cache=self.deterministic_cache,
                    seed=next(self._bulk_random),
                )
                match.play()
                outcome.append(match.state_distribution())
                match_scores = match.final_score_per_turn()
                scores[i] += match_scores[0]
                scores[j] += match_scores[1]
            self.score_history.append(scores)
            outcomes_per_round = sum(outcome, Counter())
            self.outcomes_per_round[self.round] = outcomes_per_round
            return scores

    """Axelrod MoranProcess class """
    def __next__(self):
         set_player_attributes(self.players, distributions[sys.argv[1]], float(sys.argv[2]))
         super().__next__()
         return self


if __name__ == "__main__":
    main()