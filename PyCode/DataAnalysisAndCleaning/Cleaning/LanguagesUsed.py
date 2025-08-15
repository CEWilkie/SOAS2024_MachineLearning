import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 20)

df = pd.read_csv("../../2024SODeveloperSurvey/survey_results_public.csv")

languagesDF = pd.DataFrame()
languagesDF["LanguageHaveWorkedWith"] = df["LanguageHaveWorkedWith"]
languagesDF["LanguageWantToWorkWith"] = df["LanguageWantToWorkWith"]
languagesDF["LanguageAdmired"] = df["LanguageAdmired"]

# drop null responses
languagesDF = languagesDF[languagesDF.isnull().sum(axis=1) < 3]
languagesDF.fillna(0)

# create dictionary from languages
# first get list of all languages
langList = []
for record in languagesDF.iterrows():
    for idx in range(2):
        hwwStr = str(record[1].iloc[idx])
        if hwwStr == "0":
            continue

        langs = hwwStr.split(';')
        for lang in langs:
            if lang in langList:
                continue
            else:
                langList.append(lang)

# then compose into a dict
langDict = {lang:(x+1) for x, lang in enumerate(langList)}
print(langDict)

# now create a new df using int vars
numericalDF = languagesDF.copy()
for idx in numericalDF.count():
    hwwStr = str(numericalDF.iloc[0].iloc[0])

    langs = hwwStr.split(';')
    for lang in langs:
        if lang in langList:
            continue
        else:
            langList.append(lang)

    break
print(numericalDF)