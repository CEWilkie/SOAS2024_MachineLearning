import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm

originalDF = pd.read_csv("../../2024SODeveloperSurvey/survey_results_public.csv")

# Determine the median value, and total amount of people greater than median
CompDF = originalDF.copy()
medianComp = CompDF['ConvertedCompYearly'].median()
CompDF['ConvertedCompYearly'].fillna(0, inplace=True)
countOverMedian = CompDF[CompDF['ConvertedCompYearly'] > medianComp]['ConvertedCompYearly'].count()

print(f'{medianComp} count {countOverMedian}')
print("min and max:\n", CompDF['ConvertedCompYearly'].sort_values())


# Determine quantile values above/below which outliers are removed
ccy = originalDF['ConvertedCompYearly'].copy()
ccy = ccy.dropna()

quantiles = ccy.quantile([0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975])
print(quantiles)

# new series to use suitable values
quantileCCY = ccy[ccy.between(quantiles[0.025], quantiles[0.975])]

# Save the updated csv
updatedDF = originalDF.copy()
updatedDF['ConvertedCompYearly'] = quantileCCY
updatedDF = updatedDF.dropna(subset=['ConvertedCompYearly'])
print(f"Remaining are {updatedDF['ConvertedCompYearly'].count()} valid responses")
updatedDF.to_csv("../../2024SODeveloperSurvey/CompCleanedResultsSet.csv")

# plot the before and after of the cleaning process
fig, (plotA, plotB) = plt.subplots(1, 2)
plotA.boxplot(ccy, labels=['ORIGINAL'])
plotB.boxplot(quantileCCY, labels=['CLEANED'], whis=[0, 100])
fig.suptitle("Converted Comp Yearly: Before and After Cleaning")
plt.show()

# Determine if the median or count has changed post cleaning
newMedian = quantileCCY.median()
newCountOverMedian = quantileCCY[quantileCCY > medianComp].count()
print(f"{newMedian} count {newCountOverMedian}")

# store the values
# quantileCCY.to_csv("../2024SODeveloperSurvey/cleanedAnnualCompensation.csv")

# Create Normal Distribution of data with histogram
mu, std = norm.fit(quantileCCY)
print(mu, std)
plt.hist(quantileCCY, bins=10, density=True)

x = np.linspace(quantileCCY.min(), quantileCCY.max(), 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p)
plt.show()
