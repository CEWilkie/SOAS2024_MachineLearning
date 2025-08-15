import pandas as pd

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 220)
pd.set_option('display.max_colwidth', 30)

reducedDF = pd.read_csv("../../ModifiedData/DataCleaning/SelectFieldsSet.csv")
transformedData = reducedDF[["Employment"]].copy()

# Split Employment into separate columns, merge not working roles
employmentCols = transformedData["Employment"].str.get_dummies(sep=";")
employmentCols["Not employed"] = (employmentCols["Not employed, and not looking for work"] +
                                  employmentCols["Not employed, but looking for work"] +
                                  employmentCols["Retired"])
employmentCols = employmentCols.drop(columns=["Not employed, and not looking for work", "Not employed, but looking for work", "Retired"])
employmentCols[employmentCols["Not employed"] > 1] = 1

transformedData = transformedData.drop(columns=["Employment"])
transformedData = pd.concat([transformedData, employmentCols], axis=1)


# Embedded Systems


# Merge with OrdinalData
ordinalData = pd.read_csv("../../ModifiedData/TransformedData/OrdinalData.csv")
transformedData = pd.concat([ordinalData, transformedData], axis=1)
transformedData.to_csv("CleanedData.csv", index=False)