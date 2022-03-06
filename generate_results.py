import csv
import os
import sys
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style("ticks")


def main():
    
    # check if the script was called correctly 
    if len(sys.argv) < 3:
        print("Please specify what population the agents should be drawn from and what independence should be given to PLAYERS.")
        return False  

    directory_in_str = "results/outcomes_per_round/"

    with open("results/outcomes_per_round/summary_results_mass_" + sys.argv[1] + "_independence_" + sys.argv[2] + ".csv", 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round','seed','strategy','count','percentage'])
        
        for i, file in enumerate(os.listdir(directory_in_str)):
            filename = os.fsdecode(file)
            if filename.endswith(str(sys.argv[1]) + "_independence_" + str(sys.argv[2]) + "_outcomes.csv"): 
                df_final = pd.DataFrame()
                outcomes_csv = os.path.join(directory_in_str, filename)
                
                df = pd.read_csv(outcomes_csv)
                df['seed'] = int(re.sub('[^0-9]','',outcomes_csv))
                df['exploitation'] = df['exploit'] + df['exploit_']
                df['total'] = df['coop'] + df['defect'] + df['exploitation'] 
                df['coop_percentage'] = round(df['coop'] / df['total'],2)
                df['defect_percentage'] = round(df['defect'] / df['total'],2)
                df['exploitation_percentage'] = round(df['exploitation'] / df['total'],2)
                
                df_plot = df[['coop','defect','exploitation']]
                df_plot = sns.lineplot(data=df_plot, linewidth = 2.5)
                sns.move_legend(df_plot, "upper right", bbox_to_anchor=(.99, .99), title='outcome')
                df_plot.set_title('Outcomes of simulation.' + ' Mass: ' + sys.argv[1] + ' Independence: ' + sys.argv[2] + ' Seed: ' + re.sub('[^0-9]','',outcomes_csv))
                df_plot.set_ylabel('count')
                df_plot.set_xlabel('round')
                fig = df_plot.get_figure()
                fig.savefig("results/outcomes_per_round/seed_" + re.sub('[^0-9]','',outcomes_csv) + ".png")
                plt.clf()

                df = pd.melt(df, id_vars=['round','seed'], value_vars=['coop','defect','exploitation'])
                df['percentage'] = round(df['value'] / df.groupby(['seed','round'])['value'].transform('sum'),2)
                df = df.sort_values(by=['round','seed'])
                for row in df.values.tolist():
                    writer.writerow(row)
            else:
                continue

    df = pd.read_csv("results/outcomes_per_round/summary_results_mass_" + sys.argv[1] + "_independence_" + sys.argv[2] + ".csv")
    df['average_percentage'] = round(df.groupby(['seed','strategy'])['percentage'].transform('mean'),3)
    df['sd_percentage'] = round(df.groupby(['seed','strategy'])['percentage'].transform('std'),3)  
    df['average_sum'] = round(df.groupby(['seed','strategy'])['count'].transform('sum'),0) 
    df['sd_sum'] = df.groupby(['seed','strategy'])['count'].transform('std').astype(int)
    df = df[['seed','strategy','average_sum','sd_sum','average_percentage','sd_percentage']]
    df = df.drop_duplicates()

    df_plot = sns.barplot(x = "seed", y = "average_percentage", hue="strategy",data = df)
    sns.move_legend(df_plot, "upper right", bbox_to_anchor=(1, 1), title='outcome')
    df_plot.set_title('Outcomes of simulation.' + ' Mass: ' + sys.argv[1] + ', Independence: ' + sys.argv[2])
    df_plot.set_ylabel('average percentage')
    df_plot.set_xlabel('seed')
    fig = df_plot.get_figure()
    fig.savefig("results/outcomes_per_round/mass_ " + sys.argv[1] + "_independence_" + sys.argv[2] + "average_percentage.png")
    plt.clf()

    df.to_csv("results/outcomes_per_round/summary_results_mass_" + sys.argv[1] + "_independence_" + sys.argv[2] + "_averages.csv")


if __name__ == "__main__":
    main()