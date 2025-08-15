import pandas as pd
from category_encoders import CountEncoder

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 220)
pd.set_option('display.max_colwidth', 30)

reducedDF = pd.read_csv("../../ModifiedData/DataCleaning/SelectFieldsSet.csv")
frequencyEncodedData = reducedDF[["DevType", "Country", "Industry"]].copy()

# Encoding Dev Type
encoderDev = CountEncoder()
frequencyEncodedData["DevType"] = encoderDev.fit_transform(frequencyEncodedData["DevType"])
print(frequencyEncodedData["DevType"])

# Encoding Industry
encoderDev = CountEncoder()
frequencyEncodedData["Industry"] = encoderDev.fit_transform(frequencyEncodedData["Industry"])
print(frequencyEncodedData["Industry"])

# Encoding Country
encoderDev = CountEncoder()
frequencyEncodedData["Country"] = encoderDev.fit_transform(frequencyEncodedData["Country"])
print(frequencyEncodedData["Country"])

# Merge cleanedData, labels
cleanedData = pd.read_csv("../../ModifiedData/TransformedData/CleanedData.csv")
cleanedData = pd.concat([cleanedData, frequencyEncodedData], axis=1)

cleanedData.to_csv("CleanedDataWithFrequency.csv", index=False)