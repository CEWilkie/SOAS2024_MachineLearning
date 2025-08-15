import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

originalDF = pd.read_csv("../../2024SODeveloperSurvey/survey_results_public.csv")

# Amount of null fields to clean
nullsponses = originalDF.isnull().sum()
print(f"{nullsponses.sum()} total NA Values")

x = (nullsponses / 65447) * 100
print(x)


cleanedDF = pd.read_csv("../../ModifiedData/CompCleanedResultsSet.csv")
cleanedNullsponses = cleanedDF.isnull().sum()
print(f"{cleanedNullsponses.sum()} total NA Values")
print(f"{len(cleanedDF.columns)} columns")

x = (cleanedNullsponses / 65447) * 100
print(x)

print(f"total reduction = {(cleanedNullsponses.sum() / nullsponses.sum()) * 100}%")
print(f"mean NA values {x.mean()}")


cleanedAndReducedDF = pd.read_csv("../../ModifiedData/DataCleaning/SelectFieldsSet.csv")
cleanedReducedNullsponses = cleanedAndReducedDF.isnull().sum()
print(f"{cleanedReducedNullsponses.sum()} total NA Values")
print(f"{len(cleanedAndReducedDF.columns)} columns")

nullColumnsDict = {}
for col in cleanedAndReducedDF.columns:
    naCount = cleanedAndReducedDF[col].isnull().sum()
    if naCount == 0: continue
    nullColumnsDict[col]=naCount


import textwrap
def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=90)


import seaborn as sns
sns.set_theme(style='darkgrid', font_scale=0.7)

fig, ax = plt.subplots()
ax.bar(range(len(nullColumnsDict)), list(nullColumnsDict.values()), align='center')
ax.set_xticks(np.arange(len(nullColumnsDict.keys())))
ax.set_xticklabels(list(nullColumnsDict.keys()))
wrap_labels(ax, 12, True)
plt.show()