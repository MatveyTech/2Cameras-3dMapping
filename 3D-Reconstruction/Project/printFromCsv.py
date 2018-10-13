import plotlywrapper as pw
import pandas as pd
#plot = pw.line(x=[1, 2, 3, 4], y=[4, 1, 3, 7])
df = pd.read_csv('example.csv')
#dumb,x,y,z,color =df.T.T
x = df[df.columns[1]]
y = df[df.columns[2]]
z = df[df.columns[3]]
color =df[df.columns[4]]
colors = []
counter =0
for col in color:
    colors.append(str(col))
    counter =counter +1

#todo fix coloring
#print(str(color[0]))
plot = pw.scatter3d(x,y,z,None,colors)
plot.show()