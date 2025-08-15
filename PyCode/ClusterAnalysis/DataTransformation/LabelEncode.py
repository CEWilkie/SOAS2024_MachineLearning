import pandas as pd
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 220)
pd.set_option('display.max_colwidth', 30)

reducedDF = pd.read_csv("../../ModifiedData/DataCleaning/SelectFieldsSet.csv")
labelEncodedData = reducedDF[["DevType", "Country", "Industry"]].copy()

# Encoding Dev Type
encoderDev = LabelEncoder()
labelEncodedData["DevType"] = encoderDev.fit_transform(labelEncodedData["DevType"])
print(encoderDev.classes_)


# Encoding Industry
encoderInd = LabelEncoder()
labelEncodedData["Industry"] = encoderInd.fit_transform(labelEncodedData["Industry"])
print(encoderInd.classes_)


# Encoding Country
encoderCou = LabelEncoder()
labelEncodedData["Country"] = encoderCou.fit_transform(labelEncodedData["Country"])
print(encoderCou.classes_)


# Merge cleanedData, labels
cleanedData = pd.read_csv("../../ModifiedData/TransformedData/CleanedData.csv")
cleanedData = pd.concat([cleanedData, labelEncodedData], axis=1)

cleanedData.to_csv("CleanedDataWithLabels.csv", index=False)