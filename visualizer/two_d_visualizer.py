from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from matplotlib.colors import TwoSlopeNorm
from base.visualizer import Visualizer 


class TwoDimensionVisualizer(Visualizer):
    
    def __init__(self, ws_to_coors, data, plot_type = ''):
        super().__init__(data, plot_type)
        self.ws_to_coors = ws_to_coors

    def get_coordinates(self, data, ws):
        """_summary_

        Args:
            data (_type_): _description_
            ignore_index (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        all_xs = []
        all_ys = []
        all_values =  []
        for d in data:
            v = d["w"]
            values = d["value"]
            all_xs.append(values[0])
            all_ys.append(values[1])
            all_values.append(float(v)) 
        return all_xs, all_ys, all_values


    def draw(self, data, ws, metric_name):
        """_summary_

        Returns:
            _type_: _description_
        """

        plt.rc('legend',fontsize=50)
        # xs, ys, values = self.get_coordinates(data[metric_name], ws)

        color_dict = {
            "0.2": "red",
            "0.3": "orange",
            "mmcts": "blue",
            "unimind": "gray",
            "color": "brown",
            "bert": "green"

        }

      
        sns.set_style("whitegrid")

        # create a big figure with multiple sub-figures
        # a figure with one row and two columns
        fig, axes = plt.subplots(figsize=(8, 8))
        
        # tcp red
        # rtcp green
        # mmcts blue
        # df = pd.DataFrame(data["SR"])
        # print(df)

        # # # create hisplot for target frequency
        # sns.barplot(df,
        #              x = 'w',
        #              y = 'value',
        #              color="cyan", 
        #              ax=axes[0],
        #              )

        # axes[0].set_xticklabels(axes[0].get_xticks(), size=24)
        # axes[0].set_yticklabels(axes[0].get_yticks(), size=24)

        # axes[0].set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], size=24)
        # # axes[0].xaxis.set_major_formatter(FormatStrFormatter('%d'))

        # axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # axes[0].set_ylabel("SR", fontsize=24)
        # axes[0].set_xlabel("w_gain", fontsize=24)

        # df = pd.DataFrame(data["Rewards"])
        # print(df)

        # print(xs, ys)
        # create lineplot for relative sr
        sns.lineplot(data = df,
                    x='R_gain', 
                    y='R_fair',
                    dashes=True, 
                    linestyle='--',
                    ax=axes[1],
                    linewidth=5,
                    palette=["red"],
                    alpha=1)
        

        sns.scatterplot(
            data=df,
            x= 'R_gain',
            y='R_fair',
            hue='w',
            style='w',
            sizes = [250, 250, 250, 250, 250],
            s=1500,
            markers=['*'],
            ax=axes[1],

        )

        axes[1].axhline(y=0.2554, linewidth=2, color='red', ls=':')
        axes[1].axvline(x=0.9544, linewidth=2, color='red', ls=':')

        
        # axes[0].get_legend().remove()
        axes[1].set_ylabel("R_Fair", fontsize=24)
        axes[1].set_xlabel("R_Gain", fontsize=24)

        axes[1].set_yticklabels(axes[1].get_yticks(), size=20)
        axes[1].set_xticklabels(axes[1].get_xticks(), size=20)

        # axes[1].set_xticklabels([1, 3, 5, 7, 9], size=24)
        axes[1].set(ylim=(0.0, 0.40))
        axes[1].set(xlim=(0.48, 1.0))
        axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        
        axes[1].get_legend().remove()

        # axes.legend(title="w_gain", fontsize = 24, loc="upper right", prop={'size': 20})
        # plt.setp(axes.get_legend().get_title(), fontsize='24') # for legend title

        lines_labels = [axes[1].get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc='lower center', ncol=8, prop={'size': 24})

        fig = fig.get_figure()
        fig.savefig(f"results/images/adapt_gain_fair.png", bbox_inches='tight')
