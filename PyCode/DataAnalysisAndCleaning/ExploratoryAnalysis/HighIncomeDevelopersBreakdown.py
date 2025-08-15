import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid', font_scale=0.7)

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 220)
pd.set_option('display.max_colwidth', 30)

df = pd.read_csv("../../ModifiedData/DataCleaning/SelectFieldsSet.csv")

df["IsHighIncome"] = df["ConvertedCompYearly"] > 65000
print(df["IsHighIncome"].value_counts())
print(df["ConvertedCompYearly"].median())

ax = df["IsHighIncome"].value_counts().plot(kind="bar", title="High-Income Developers", xlabel="Has Over 65,000 in Annual Compensation")
ax.bar_label(ax.containers[0], label_type="edge")
plt.show()