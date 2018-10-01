from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

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
    
    #print(final_list)
    return final_list

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')

sss = np.load("testout77.npy")
print (type(sss))
print (sss.shape)
#for s in sss:
#    print(s)
#sss = RemoveOutliers(sss)
#print (len(sss))
#print (sss.shape)

number = 0
colors = ['red','green','blue','pink']
color = 'red'
for s in sss:
    number=number+1
    ax.scatter(s[0], s[1], s[2],marker='.', color = colors[number % 4], s=40, label='class 1')
    text = str(int(s[0])) + ', ' + str(int(s[1])) + ', ' + str(int(s[2]))
    #ax.text(int(s[0]), int(s[1]), int(s[2]), text, zdir=(1, 1, 1))
    #text = str(int(s[0])) + ', ' + str(int(s[1])) + ', ' + str(int(s[2]))
    

ax.set_xlabel(' X')
ax.set_ylabel(' Y')
ax.set_zlabel(' Z')


set_axes_equal(ax)
plt.show()