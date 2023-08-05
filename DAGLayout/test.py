import numpy as np
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
pd.options.display.notebook_repr_html = False 
plt.rcParams['figure.dpi'] = 75 
sns.set_theme(style='darkgrid') 

from DAGLayout import DAGLayout

#data = pickle.load(open("/tmp/VideoVis/weightlifting/2022-03-17 13:46:48.pkl", 'rb'))
#data = pickle.load(open("/tmp/VideoVis/basketball/2022-03-17 13:58:45.pkl", 'rb'))
#data = pickle.load(open("/tmp/VideoVis/weightlifting/2022-03-17 13:45:49.pkl", 'rb'))

#data = pickle.load(open("/tmp/VideoVis/2022-03-20 23:09:56.pkl", 'rb'))
#data = pickle.load(open("/tmp/VideoVis/2022-03-21 00:09:59.pkl", 'rb'))
data = pickle.load(open("/tmp/VideoVis/2022-03-21 11:51:45.pkl", 'rb'))

dist, valid, labels = (data['D'], data['A'], data['labels'])
dist = np.array(dist)
valid = np.array(valid)
labels = np.array(labels)

y = DAGLayout(dist, valid, labels, 400, 0.4, 15)
y = 1 - np.array(y)

data = pickle.load(open("/tmp/VideoVis/2022-03-21 13:02:57_layout.pkl", 'rb'))
dist, valid, labels = (data['D'], data['A'], data['labels'])
y = np.array(data['y'])

points = []
for i in range(y.shape[0]):
    curr_valid = np.flatnonzero(valid[i] > 0)
    #for j in range(y.shape[1]):
    for j in range(curr_valid[0], curr_valid[-1] + 1):
        points.append([j, i, y[i, j], labels[i]])
points = np.array(points)
df = pd.DataFrame({ 'x': points[:, 0], 'line': points[:, 1], 'y': points[:, 2], 'label': points[:, 3] })
plt.figure(figsize=(15, 10))
sns.lineplot(data=df, x='x', y='y', hue='label', style='line', markers=True)
plt.savefig('1.png')