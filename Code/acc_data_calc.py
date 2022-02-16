import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

x = np.random.normal(loc=1.5, scale=0.6, size=1000)
y = np.random.uniform(-10,10,200)

z = np.concatenate((x, y))

with open('./acc_data.csv', 'w', encoding='UTF8', newline='') as file_open:
    writer = csv.writer(file_open)
    for i in range(0, len(z)):
        writer.writerow([i, z[i]])

sns.distplot(z)
plt.show()