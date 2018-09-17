from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt

# Generate some 3D sample data
mu_vec1 = np.array([0,0,0]) # mean vector
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]]) # covariance matrix

sss = np.load("testout77.npy")
#sss[:,2]=0
#sss[:,1]=0
#sss[:,0]=0
# class1_sample.shape -> (20, 3), 20 rows, 3 columns

fig = plt.figure(figsize=(16,16))
ax = fig.add_subplot(111, projection='3d')
print(sss)
for s in sss:
    ax.scatter(s[0], s[1], s[2],marker='.', color='red', s=40, label='class 1')
    text = str(int(s[0])) + ', ' + str(int(s[1])) + ', ' + str(int(s[2]))
    ax.text(int(s[0]), int(s[1]), int(s[2]), text, zdir=(1, 1, 1))
#ax.scatter(class1_sample[:,0], class1_sample[:,1], class1_sample[:,2], 
#   marker='x', color='blue', s=40, label='class 1')
#ax.scatter(class2_sample[:,0], class2_sample[:,1], class2_sample[:,2], 
 #          marker='o', color='green', s=40, label='class 2')
#ax.scatter(class3_sample[:,0], class3_sample[:,1], class3_sample[:,2], 
#           marker='^', color='red', s=40, label='class 3')

ax.set_xlabel('variable X')
ax.set_ylabel('variable Y')
ax.set_zlabel('variable Z')

plt.title('3D Scatter Plot')
     
plt.show()