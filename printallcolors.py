#print all scences together with diffrent colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
def RemoveOutliers(points_list):

    data = np.array(points_list)
    # split to x,y,z 
    x_pos, y_pos, z_pos = data.T
    #points_list = x;
    #elements = np.array(points_list)

    mean_x = np.mean(x_pos, axis=0)
    mean_y = np.mean(y_pos, axis=0)
    mean_z = np.mean(z_pos, axis=0)
    
    sd_x = np.std(x_pos, axis=0)
    sd_y = np.std(y_pos, axis=0)
    sd_z = np.std(z_pos, axis=0)

    final_list = [x for x in points_list if (x[0] > mean_x - 2 * sd_x)]
    final_list = [x for x in final_list if (x[0] < mean_x + 2 * sd_x)]

    final_list = [y for y in final_list if (y[1] > mean_y - 2 * sd_y)]
    final_list = [y for y in final_list if (y[1] < mean_y + 2 * sd_y)]

    final_list = [z for z in final_list if (z[2] > mean_z - 2 * sd_z)]
    final_list = [z for z in final_list if (z[2] < mean_z + 2 * sd_z)]
    
    print(final_list)
    return final_list

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')


#for s in sss:
#    print(s)

number = 0
colors = ['grey','pink','red','orange','yellow','purple','green','blue']
color = 'red'

pathToOutputs = "C:/matvery/2Cameras-3dMapping/output/"

print("Printing points from folder: \n"+pathToOutputs)
allPoints = []
# setting path to 3d points
plots = glob.glob(pathToOutputs+"*.npy")
for plot in plots:
    number = number + 1
    currentColor = colors[number % 8]
    points = np.load(plot)
    
    
    points = RemoveOutliers(points)
  
    first = True
    for point in points:
        newpoint=[[point[0],point[1],point[2],currentColor]]
        allPoints = allPoints +newpoint
        ax.scatter(point[0], point[1], point[2],marker='.', color = currentColor, s=40, label='class 1')
        text = str(int(point[0])) + ', ' + str(int(point[1])) + ', ' + str(int(point[2]))
        if first is True:
            ax.text(int(point[0]), int(point[1]), int(point[2]), number, zdir=(1, 1, 1))
            first = False
        #ax.text(int(s[0]), int(s[1]), int(s[2]), text, zdir=(1, 1, 1))
        #text = str(int(s[0])) + ', ' + str(int(s[1])) + ', ' + str(int(s[2]))
        
ax.set_xlabel(' X')
ax.set_ylabel(' Y')
ax.set_zlabel(' Z')
#np.savetxt("foo.csv", allPoints, delimiter=",",format='%10.5f')
df=pd.DataFrame(allPoints, columns = ['x', 'y', 'z', 'color'])
df.to_csv('example.csv')
set_axes_equal(ax)
plt.show()