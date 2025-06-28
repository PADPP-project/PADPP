from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

from base.visualizer import Visualizer 


class ThreeDimensionVisualizer(Visualizer):
    
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
            coor = d["coor"]
            v = d["value"]
            all_xs.append(float(coor[self.ws_to_coors[ws[0]]]))
            all_ys.append(float(coor[self.ws_to_coors[ws[1]]]))
            all_values.append(float(v)) 
        return all_xs, all_ys, all_values


    def draw(self, data, ws, metric_name):
        """_summary_

        Returns:
            _type_: _description_
        """
        x, y, z = self.get_coordinates(data[metric_name], ws)

        # Creating figure
        fig = plt.figure(figsize =(6, 6)) 
        ax = plt.axes(projection ='3d') 

        # Creating color map
        my_cmap = plt.get_cmap('Oranges')

        for x1, y1, z1 in list(zip(x,y,z)):
            # ax.bar([x1], y1, dx =0.1, dy = 0.1)
            ax.bar3d(x1, y1, 0, dx=0.005, dy = 0.05, dz = z1, color = 'red')
            # ax.bar3d(x1, 0, 0, dx=0.05, dy = 0.000, dz = z1, color='red')

            ax.scatter(x1, y1, z1, marker='*', s = 200)
            ax.scatter(x1, y1, 0, marker='x', s = 30)


        # Creating plot
        trisurf = ax.plot(x, y, 0,
                                linewidth = 1, 
                                linestyle='--',
                                ) 
        
        trisurf = ax.plot(x, y, z,
                                linewidth = 3, 
                                linestyle='--',
                                ) 
        # ax.bar3d(x, y, z, zdir='z', offset=-100, cmap='coolwarm')


        # fig.colorbar(trisurf, ax = ax, shrink = 0.5, aspect = 5)
        ax.set_title(f"Trade-off Between {ws[0]} and {ws[1]} on {metric_name}")

        ax.azim = 60
        ax.elev = 25
        ax.dist = 10

        x_label = ws[0]
        y_label = ws[1]

        # Adding labels
        ax.set_xlabel(x_label, fontweight ='bold') 
        ax.set_ylabel(y_label, fontweight ='bold') 
        ax.set_zlabel(metric_name, fontweight ='bold')
            
        # show plot
        plt.subplots_adjust(left=0.35,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)


        fig = fig.get_figure()
        fig.savefig(f"results/images/rd_gain_fair.pdf", bbox_inches='tight')
