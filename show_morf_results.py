import pandas as pd
from matplotlib import pyplot as plt

df1= pd.read_csv('results/df_res_gradcam_400.csv').T[1]
df2= pd.read_csv('results/df_res_felz_055_500.csv').T[1]
df3= pd.read_csv('results/df_res_quick_055_700.csv').T[1]
df4 = pd.read_csv('results/df_res_slico_055_250.csv').T[1]
print(df1.sum())
print(df2.sum())
print(df3.sum())
print(df4.sum())
df1.cumsum().plot()
df2.cumsum().plot()
df3.cumsum().plot()
df4.cumsum().plot()
plt.show()

