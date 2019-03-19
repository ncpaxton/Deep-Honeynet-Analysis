import pandas as pd
from matplotlib import pyplot

df = pd.read_csv('validation_losses.csv')
df = df.drop('Unnamed: 0', axis=1)
df.plot()
pyplot.show()
