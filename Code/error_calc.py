import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

np.random.seed(0)
x = np.random.normal(loc=1.5, scale=0.6, size=1000)
y = np.random.uniform(-10,10,20)

z = np.concatenate((x, y))
sns.distplot(z)
plt.show()