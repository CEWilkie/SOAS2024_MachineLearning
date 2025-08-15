import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../../ModifiedData/DataCleaning/SelectFieldsSet.csv")

x = df['Industry']
y = df['YearsWorkExperience']

col = np.where(df["ConvertedCompYearly"]>65000,'#00b579','b')
fig, ax = plt.subplots()
ax.scatter(x, y, c=col,alpha=0.5)

import textwrap
def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=90)

wrap_labels(ax, 10)
plt.xticks(rotation=90)
plt.xlabel("Industry")
plt.ylabel("Years of Work Experience")
plt.show()
