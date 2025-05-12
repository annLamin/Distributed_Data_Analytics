import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('cluster_data.csv', header=0, index_col=0)
print(data.head(10))

scatter_data = data[['x', 'y']].values
plt.scatter(scatter_data[:, 0], scatter_data[:, 1], s=10)
plt.show()