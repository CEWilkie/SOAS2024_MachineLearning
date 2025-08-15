import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 220)
pd.set_option('display.max_colwidth', 30)

cleanOriginalDF = pd.read_csv("../../ModifiedData/CompCleanedResultsSet.csv")

# All Data Columns
# 'ResponseId', 'MainBranch', 'Age', 'Employment', 'RemoteWork', 'Check', 'CodingActivities', 'EdLevel', 'LearnCode',
# 'LearnCodeOnline', 'TechDoc', 'YearsCode', 'YearsCodePro', 'DevType', 'OrgSize', 'PurchaseInfluence', 'BuyNewTool',
# 'BuildvsBuy', 'TechEndorse', 'Country', 'Currency', 'CompTotal', 'LanguageHaveWorkedWith', 'LanguageWantToWorkWith',
# 'LanguageAdmired', 'DatabaseHaveWorkedWith', 'DatabaseWantToWorkWith', 'DatabaseAdmired', 'PlatformHaveWorkedWith',
# 'PlatformWantToWorkWith', 'PlatformAdmired', 'WebframeHaveWorkedWith', 'WebframeWantToWorkWith', 'WebframeAdmired',
# 'EmbeddedHaveWorkedWith', 'EmbeddedWantToWorkWith', 'EmbeddedAdmired', 'MiscTechHaveWorkedWith', 'MiscTechWantToWorkWith',
# 'MiscTechAdmired', 'ToolsTechHaveWorkedWith', 'ToolsTechWantToWorkWith', 'ToolsTechAdmired', 'NEWCollabToolsHaveWorkedWith',
# 'NEWCollabToolsWantToWorkWith', 'NEWCollabToolsAdmired', 'OpSysPersonal use', 'OpSysProfessional use',
# 'OfficeStackAsyncHaveWorkedWith', 'OfficeStackAsyncWantToWorkWith', 'OfficeStackAsyncAdmired', 'OfficeStackSyncHaveWorkedWith',
# 'OfficeStackSyncWantToWorkWith', 'OfficeStackSyncAdmired', 'AISearchDevHaveWorkedWith', 'AISearchDevWantToWorkWith',
# 'AISearchDevAdmired', 'NEWSOSites', 'SOVisitFreq', 'SOAccount', 'SOPartFreq', 'SOHow', 'SOComm', 'AISelect', 'AISent',
# 'AIBen', 'AIAcc', 'AIComplex', 'AIToolCurrently Using', 'AIToolInterested in Using', 'AIToolNot interested in Using',
# 'AINextMuch more integrated', 'AINextNo change', 'AINextMore integrated', 'AINextLess integrated', 'AINextMuch less integrated',
# 'AIThreat', 'AIEthics', 'AIChallenges', 'TBranch', 'ICorPM', 'WorkExp', 'Knowledge_1', 'Knowledge_2', 'Knowledge_3',
# 'Knowledge_4', 'Knowledge_5', 'Knowledge_6', 'Knowledge_7', 'Knowledge_8', 'Knowledge_9', 'Frequency_1', 'Frequency_2',
# 'Frequency_3', 'TimeSearching', 'TimeAnswering', 'Frustration', 'ProfessionalTech', 'ProfessionalCloud', 'ProfessionalQuestion',
# 'Industry', 'JobSatPoints_1', 'JobSatPoints_4', 'JobSatPoints_5', 'JobSatPoints_6', 'JobSatPoints_7', 'JobSatPoints_8',
# 'JobSatPoints_9', 'JobSatPoints_10', 'JobSatPoints_11', 'SurveyLength', 'SurveyEase', 'ConvertedCompYearly', 'JobSat'


# Specify Headers to retain
retainHeaders = ['ResponseID', 'MainBranch', 'Age', 'Employment', 'RemoteWork', 'CodingActivities', 'EdLevel',
                 'LearnCode', 'LearnCodeOnline', 'TechDoc',

                 'YearsCode', 'YearsCodePro', 'DevType', 'OrgSize', 'PurchaseInfluence',
                 'Country', 'Currency', 'ConvertedCompYearly',

                 'LanguageHaveWorkedWith',
                 'DatabaseHaveWorkedWith',
                 'PlatformHaveWorkedWith',
                 'WebframeHaveWorkedWith',
                 'EmbeddedHaveWorkedWith',
                 'MiscTechHaveWorkedWith',
                 'ToolsTechHaveWorkedWith',
                 'NEWCollabToolsHaveWorkedWith',
                 'OpSysPersonal use', 'OpSysProfessional use',
                 'OfficeStackAsyncHaveWorkedWith',
                 'OfficeStackSyncHaveWorkedWith',
                 'AISearchDevHaveWorkedWith',

                 'SOVisitFreq', 'SOPartFreq', 'SOHow',

                 'AISelect', 'AISent', 'AIThreat',

                 'ICorPM', 'WorkExp', 'TimeSearching', 'TimeHelping', 'ProfessionalTech', 'Industry']

# Produce Dictionary conversions from Original Header to New Header
headerDict = {
    'TechDoc':'TechnicalDocumentationUsage',
    'WorkExp':'YearsWorkExperience',
    'NEWCollabToolsHaveWorkedWith':'DevelopmentEnvironments',
    'OpSysPersonal use': 'PersonalOperatingSystem',
    'OpSysProfessional use': 'WorkOperatingSystem',
    'OfficeStackAsyncHaveWorkedWith':'ManagementTools',
    'OfficeStackSyncHaveWorkedWith': 'CommunicationTools',
    'LanguageHaveWorkedWith':'RecentLanguages',
    'DatabaseHaveWorkedWith':'RecentDatabases',
    'PlatformHaveWorkedWith':'RecentCloudPlatforms',
    'WebframeHaveWorkedWith':'RecentWebTech',
    'EmbeddedHaveWorkedWith':'RecentEmbeddedSystems',
    'MiscTechHaveWorkedWith':'RecentOtherFrameworks',
    'ToolsTechHaveWorkedWith':'RecentDevelopmentTools',
    'AISearchDevHaveWorkedWith':'RecentAITools',

    'SOVisitFreq':'FrequencyOfVisitingSO',
    'SOPartFreq':'FrequencyOfParticipatingSO',
    'SOHow':'HowDoYouUseSO',

    'AISelect':'AITools',
    'AISent':'StanceOnAITools',
    'AIThreat':'DoesAIThreatenYourJob',

    'ICorPM':'IndividualOrManager',
    'CompanyTech':'ProfessionalTech'
}

# Now compile multi-headers into single header of key info
# and apply new Headers
condensedDF = cleanOriginalDF[cleanOriginalDF.columns.intersection(retainHeaders)]
condensedDF = condensedDF.rename(columns=headerDict)
# print(condensedDF)

# store updated database
condensedDF.to_csv("../../ModifiedData/SelectFieldsSetWithNull.csv")

dictDF = pd.DataFrame.from_dict({"Original Header":headerDict.keys(),"New Header":headerDict.values()})
# print(dictDF)

# Now remove null values for None in suitable fields
filledDF = condensedDF.copy()
filledDF["LearnCodeOnline"].fillna("None", inplace=True)
filledDF["TechnicalDocumentationUsage"].fillna("None", inplace=True)

filledDF["RecentLanguages"].fillna("None", inplace=True)
filledDF["RecentDatabases"].fillna("None", inplace=True)
filledDF["RecentCloudPlatforms"].fillna("None", inplace=True)
filledDF["RecentWebTech"].fillna("None", inplace=True)
filledDF["RecentEmbeddedSystems"].fillna("None", inplace=True)
filledDF["RecentOtherFrameworks"].fillna("None", inplace=True)
filledDF["RecentDevelopmentTools"].fillna("None", inplace=True)
filledDF["RecentAITools"].fillna("None", inplace=True)

filledDF["DevelopmentEnvironments"].fillna("None", inplace=True)
filledDF["PersonalOperatingSystem"].fillna("None", inplace=True)
filledDF["ManagementTools"].fillna("None", inplace=True)
filledDF["WorkOperatingSystem"].fillna("None", inplace=True)

filledDF["DevelopmentEnvironments"].fillna("None", inplace=True)
filledDF["PersonalOperatingSystem"].fillna("None", inplace=True)

filledDF.to_csv("../../ModifiedData/SelectFieldsSetWithSomeNull.csv")

# plot updated null counts

nullColumnsDict = {}
for col in filledDF.columns:
    naCount = filledDF[col].isnull().sum()
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




# For the remaining columns, fill na values with the mode response
for col in filledDF.columns:
    filledDF[col] = filledDF[col].fillna(filledDF[col].mode(dropna=True)[0])


# Drop non-developers and then the MainBranch column
counts = filledDF["MainBranch"].value_counts()
print(counts)
filledDF = filledDF.drop(filledDF[filledDF["MainBranch"] != "I am a developer by profession"].index)
filledDF = filledDF.drop(columns=["MainBranch"])

# Store updated Database
filledDF.to_csv("../../ModifiedData/SelectFieldsSet.csv")


# plot updated null counts

nullColumnsDict = {}
for col in filledDF.columns:
    naCount = filledDF[col].isnull().sum()
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

# fig, ax = plt.subplots()
# ax.bar(range(len(nullColumnsDict)), list(nullColumnsDict.values()), align='center')
# ax.set_xticks(np.arange(len(nullColumnsDict.keys())))
# ax.set_xticklabels(list(nullColumnsDict.keys()))
# wrap_labels(ax, 12, True)
# plt.show()

