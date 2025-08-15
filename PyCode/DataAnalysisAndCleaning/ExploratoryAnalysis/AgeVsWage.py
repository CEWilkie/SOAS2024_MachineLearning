import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cleanedCompData = pd.read_csv("../../ModifiedData/DataCleaning/SelectFieldsSet.csv")
print(cleanedCompData.count())

# Produce Summary of ages of respondents
ageCounts = cleanedCompData['Age'].copy().groupby(cleanedCompData['Age']).count()
print(ageCounts.index)

plt.bar(x=ageCounts.index, height=ageCounts.values)
plt.xticks(rotation=45)
for i in range(len(ageCounts.index)):
    plt.text(i, ageCounts.values[i], ageCounts.values[i], ha='center')
plt.show()


# Compare against cleaned wages
datacpy = cleanedCompData.copy()
medians = datacpy['ConvertedCompYearly'].groupby(cleanedCompData['Age']).median()

print([medians.index[i] for i in [7, 0, 1, 2, 3, 4, 5, 6]])
medians = medians.reindex(index=[medians.index[i] for i in [7, 0, 1, 2, 3, 4, 5, 6]])

plt.plot(medians)
plt.ylabel("Annual Compensation")
plt.xticks(rotation=45)
plt.show()


# Get grouped counts of above/below median
datacpy = cleanedCompData.copy()
datacpy["IsHighIncome"] = datacpy["ConvertedCompYearly"] > 65000
groupedHighIncome = datacpy.groupby("Age")["IsHighIncome"].sum()
groupedLowIncome = datacpy.groupby("Age")["IsHighIncome"].count() - groupedHighIncome
print(groupedHighIncome, groupedLowIncome)


# display as stacked bar chart
fig, ax = plt.subplots()

for group in groupedLowIncome.index:
    ax.bar(group, groupedLowIncome[group], label=("Below Median" if group==groupedLowIncome.index[0] else ""), bottom=0, color="b")

for group in groupedHighIncome.index:
    ax.bar(group, groupedHighIncome[group], label=("Above Median" if group==groupedHighIncome.index[0] else ""), bottom=groupedLowIncome[group], color='c')

ax.legend(loc="upper right")
plt.xticks(rotation=45)
plt.show()