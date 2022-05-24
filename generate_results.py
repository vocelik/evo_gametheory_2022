import csv
import os
import sys
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from vars import NUMPY_RANDOM_SEEDS

sns.set(rc={"figure.figsize": (11.7, 8.27)})
sns.set_style("ticks")


def main():

    # check if the script was called correctly
    if len(sys.argv) < 3:
        print(
            "Please specify what population the agents should be drawn from and what independence should be given to PLAYERS."
        )
        return False

    directory_in_str = "results/outcomes_per_round/"

    for population_seed in NUMPY_RANDOM_SEEDS:

        with open(
            "results/outcomes_per_round/summary_results_mass_"
            + sys.argv[1]
            + "_independence_"
            + sys.argv[2]
            + "_mass_weight_"
            + str(sys.argv[3])
            + "_independence_weight_"
            + str(sys.argv[4])
            + "_population_seed_"
            + str(population_seed)
            + ".csv",
            "w",
            encoding="UTF8",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(
                ["round", "seed", "population_seed", "strategy", "count", "percentage"]
            )

            for file in os.listdir(directory_in_str):
                filename = os.fsdecode(file)
                if filename.endswith(
                    str(sys.argv[1])
                    + "_independence_"
                    + str(sys.argv[2])
                    + "_mass_weight_"
                    + str(sys.argv[3])
                    + "_independence_weight_"
                    + str(sys.argv[4])
                    + "_outcomes_"
                    + "population_seed_"
                    + str(population_seed)
                    + ".csv"
                ):
                    outcomes_csv = os.path.join(directory_in_str, filename)
                    seed_long = re.sub("_mass.*", "", outcomes_csv)
                    seed = re.sub("[^0-9]", "", seed_long)

                    df = pd.read_csv(outcomes_csv)
                    df["seed"] = int(seed)
                    df["population_seed"] = population_seed
                    df["exploitation"] = df["exploit"] + df["exploit_"]
                    df["total"] = df["coop"] + df["defect"] + df["exploitation"]
                    df["coop_percentage"] = round(df["coop"] / df["total"], 2)
                    df["defect_percentage"] = round(df["defect"] / df["total"], 2)
                    df["exploitation_percentage"] = round(
                        df["exploitation"] / df["total"], 2
                    )

                    df_plot = df[["coop", "defect", "exploitation"]]
                    df_plot = sns.lineplot(data=df_plot, linewidth=2.5)
                    sns.move_legend(
                        df_plot,
                        "upper right",
                        bbox_to_anchor=(0.99, 0.99),
                        title="outcome",
                    )
                    df_plot.set_title(
                        "Outcomes of simulation."
                        + " Mass: "
                        + sys.argv[1]
                        + " Independence: "
                        + sys.argv[2]
                        + " Seed: "
                        + seed
                        + " Mass weight "
                        + str(sys.argv[3])
                        + " Independence weight "
                        + str(sys.argv[4])
                        + "Population seed "
                        + str(population_seed)
                    )
                    df_plot.set_ylabel("count")
                    df_plot.set_xlabel("round")
                    fig = df_plot.get_figure()
                    fig.savefig(
                        "results/outcomes_per_round/seed_"
                        + seed
                        + "_mass_"
                        + sys.argv[1]
                        + "_independence_"
                        + sys.argv[2]
                        + "_mass_weight_"
                        + str(sys.argv[3])
                        + "_independence_weight_"
                        + str(sys.argv[4])
                        + "_population_seed_"
                        + str(population_seed)
                        + ".pdf"
                    )
                    plt.clf()

                    df = pd.melt(
                        df,
                        id_vars=["round", "seed", "population_seed"],
                        value_vars=["coop", "defect", "exploitation"],
                    )
                    df["percentage"] = round(
                        df["value"]
                        / df.groupby(["seed", "round"])["value"].transform("sum"),
                        2,
                    )
                    df = df.sort_values(by=["round", "seed", "population_seed"])
                    for row in df.values.tolist():
                        writer.writerow(row)
                else:
                    continue

        df = pd.read_csv(
            "results/outcomes_per_round/summary_results_mass_"
            + sys.argv[1]
            + "_independence_"
            + sys.argv[2]
            + "_mass_weight_"
            + str(sys.argv[3])
            + "_independence_weight_"
            + str(sys.argv[4])
            + "_population_seed_"
            + str(population_seed)
            + ".csv"
        )
        df["average_percentage"] = round(
            df.groupby(["seed", "strategy"])["percentage"].transform("mean"), 3
        )
        df["sd_percentage"] = round(
            df.groupby(["seed", "strategy"])["percentage"].transform("std"), 3
        )
        df["average_sum"] = round(
            df.groupby(["seed", "strategy"])["count"].transform("sum"), 0
        )
        df["sd_sum"] = (
            df.groupby(["seed", "strategy"])["count"].transform("std").astype(int)
        )
        df = df[
            [
                "seed",
                "population_seed",
                "strategy",
                "average_sum",
                "sd_sum",
                "average_percentage",
                "sd_percentage",
            ]
        ]
        df = df.drop_duplicates()

        df_plot = sns.barplot(x="seed", y="average_percentage", hue="strategy", data=df)
        sns.move_legend(df_plot, "upper right", bbox_to_anchor=(1, 1), title="outcome")
        df_plot.set_title(
            "Outcomes of simulation."
            + " Mass: "
            + sys.argv[1]
            + ", Independence: "
            + sys.argv[2]
            + " Population seed: "
            + str(population_seed)
        )
        df_plot.set_ylabel("average percentage")
        df_plot.set_xlabel("seed")
        fig = df_plot.get_figure()
        fig.savefig(
            "results/outcomes_per_round/mass_ "
            + sys.argv[1]
            + "_independence_"
            + sys.argv[2]
            + "_mass_weight_"
            + str(sys.argv[3])
            + "_independence_weight_"
            + str(sys.argv[4])
            + "_population_seed_"
            + str(population_seed)
            + "_average_percentage.pdf"
        )
        plt.clf()

        df.to_csv(
            "results/outcomes_per_round/summary_results_mass_"
            + sys.argv[1]
            + "_independence_"
            + sys.argv[2]
            + "_mass_weight_"
            + str(sys.argv[3])
            + "_independence_weight_"
            + str(sys.argv[4])
            + "_population_seed_"
            + str(population_seed)
            + "_averages.csv"
        )


if __name__ == "__main__":
    main()
