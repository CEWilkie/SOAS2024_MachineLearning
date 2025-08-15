import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 220)
pd.set_option('display.max_colwidth', 30)

reducedDF = pd.read_csv("../../ModifiedData/DataCleaning/SelectFieldsSet.csv")
transformedData = reducedDF.copy()

# Convert Annual Comp to Above/Below Median
transformedData["HighIncomeDev"] = transformedData["ConvertedCompYearly"] > 65000
transformedData = transformedData.drop(columns=["ConvertedCompYearly"])

# Years Coding, remove "Less than 1 Year" values
transformedData["YearsCode"] = transformedData["YearsCode"].replace(to_replace="Less than 1 year", value=0)
transformedData["YearsCode"] = transformedData["YearsCode"].replace(to_replace="More than 50 years", value=50)
transformedData["YearsCodePro"] = transformedData["YearsCodePro"].replace(to_replace="Less than 1 year", value=0)
transformedData["YearsCodePro"] = transformedData["YearsCodePro"].replace(to_replace="More than 50 years", value=50)

# AGE
ages = reducedDF["Age"].value_counts().keys()
print(ages, reducedDF["Age"].value_counts())

ageDict = {'Under 18 years old': 17,
           '18-24 years old': 21,
           '25-34 years old': 29,
           '35-44 years old': 39,
           '45-54 years old': 49,
           '55-64 years old': 59,
           '65 years or older': 65,
           'Prefer not to say': 29} # only 9 respondents, use modal value

# alternatively ordinal encode?
# encodeAge = OrdinalEncoder()
# transformedData["AgeOrdinals"] = encodeAge.fit_transform(transformedData[["Age"]])

# Apply numerical dict to data
transformedData["AgeOrdinals"] = reducedDF["Age"].replace(ageDict)
print(transformedData[["Age", "AgeOrdinals"]])
print(ageDict)

# Remote
remoteGroups = transformedData["RemoteWork"].value_counts().keys().sort_values()
remoteGroups = remoteGroups.reindex([remoteGroups[i] for i in [1,0,2]])
print(remoteGroups[0].to_list())

remoteDict = {}
for i, group in enumerate(remoteGroups[0].to_list()):
    remoteDict[group] = i

# Apply numerical dict to data
transformedData["RemoteOrdinals"] = transformedData["RemoteWork"].replace(remoteDict)
print(transformedData[["RemoteWork", "RemoteOrdinals"]])
print(remoteDict)


# Education Level
educationLevels = transformedData["EdLevel"].value_counts().keys()
educationLevels = educationLevels.reindex([educationLevels[i] for i in [4, 1, 0, 5, 2, 6, 3, 7]])

educationDict = {}
for i, group in enumerate(reversed(educationLevels[0].to_list())):
    educationDict[group] = i

transformedData["EdLevelOrdinals"] = transformedData["EdLevel"].replace(educationDict)
print(transformedData[["EdLevel", "EdLevelOrdinals"]])
print(educationDict)


# Organisation Size
orgs = transformedData["OrgSize"].value_counts().keys()
print(orgs, transformedData["OrgSize"].value_counts())

orgDict = {'Just me - I am a freelancer, sole proprietor, etc.': 1,
           '2 to 9 employees': 5.5,
           '10 to 19 employees': 14.5,
           'I don’t know': 20,
           '20 to 99 employees': 59.5,
           '100 to 499 employees': 299.5,
           '500 to 999 employees': 749.5,
           '1,000 to 4,999 employees': 2999.5,
           '5,000 to 9,999 employees': 7499.5,
           '10,000 or more employees': 10000}

# alternatively ordinal encode? performs worse in clustering
# encodeOrgSize = OrdinalEncoder()
# transformedData["OrgSizeOrdinals"] = encodeOrgSize.fit_transform(transformedData[["OrgSize"]])

# Purchase Influence
influenceDict = {'I have a great deal of influence' : 2,
                 'I have some influence' : 1,
                 'I have little or no influence' : 0}
transformedData["PurchaseInfluence"] = transformedData["PurchaseInfluence"].replace(influenceDict)


# Apply ordinal dict to data
transformedData["OrgSizeOrdinals"] = transformedData["OrgSize"].replace(orgDict)
print(transformedData[["OrgSize", "OrgSizeOrdinals"]])
print(orgDict)

# Drop non ordinal originals
transformedData = transformedData[["HighIncomeDev", "AgeOrdinals", "OrgSizeOrdinals", "EdLevelOrdinals", "RemoteOrdinals", "YearsCodePro", "PurchaseInfluence"]]

# Store Dataframe
transformedData.to_csv("OrdinalData.csv", index=False)