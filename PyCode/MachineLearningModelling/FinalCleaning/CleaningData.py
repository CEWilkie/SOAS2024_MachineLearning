import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import OrdinalEncoder

# ORDINAL DATA
# EdLevel, Age, Remote, ConvertedCompYearly, YearsCode, YearsCodePro, DevType, OrgSize

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 220)
pd.set_option('display.max_colwidth', 30)

df = pd.read_csv("../../ModifiedData/TransformedData/ApplyingNominals.csv")

# Encode DevType using Frequency
freq = df["DevType"].value_counts(normalize=True)
df["DevTypeEncoded"] = df["DevType"].map(freq)
otherDevs = freq.loc[freq < 0.01].index.to_list()
retainDevs = freq.loc[freq >= 0.01].index.to_list()
devTypes = pd.get_dummies(df["DevType"])
df = pd.concat([df, devTypes], axis=1)
df["DevType_Other"] = df[otherDevs].sum()
df["DevType_Other"] = df["DevType_Other"].fillna(0)
df[df["DevType_Other"] > 1] = 1
df = df.drop(columns=otherDevs)

# Frequency Encode Work Operating System
# operatingSystems = df["WorkOperatingSystem"].str.get_dummies(sep=';')
# freq = operatingSystems.sum() / operatingSystems.count()
# df["WorkOSEncoded"] = df["WorkOperatingSystem"].map(freq)

operatingSystems = df["WorkOperatingSystem"].str.get_dummies(sep=';')
# print(df["WorkOperatingSystem"].str.get_dummies(sep=';').sum())

# Android, Arch, Debian, MacOS, OtherLinux, Ubuntu, Windows, WSL, iOS
# Merge some columns

operatingSystems["iOS"] = operatingSystems["iOS"] + operatingSystems["iPadOS"]
operatingSystems[operatingSystems["iOS"] > 1] = 1

operatingSystems["Other"] = (operatingSystems["AIX"] + operatingSystems["BSD"] + operatingSystems["ChromeOS"]
                            + operatingSystems["Cygwin"] + operatingSystems["Haiku"] + operatingSystems["Solaris"]
                             + operatingSystems["Other Linux-based"] + operatingSystems["Other (please specify):"])
operatingSystems[operatingSystems["Other"] > 1] = 1

operatingSystems["OtherLinux"] = operatingSystems["Arch"] + operatingSystems["Debian"] + operatingSystems["Red Hat"] + operatingSystems["Fedora"]
operatingSystems[operatingSystems["OtherLinux"] > 1] = 1

operatingSystems["WSL"] = operatingSystems["Windows Subsystem for Linux (WSL)"]

operatingSystems = operatingSystems[["Windows", "Android", "MacOS", "WSL", "OtherLinux", "Other"]]

# finalise data subset
numericalData = df[["HighIncomeDev", "AgeOrdinals", "EdLevelOrdinals", "OrgSizeOrdinals",
                    "RemoteOrdinals", "YearsCodePro", *retainDevs, "DevType_Other"]]
numericalData = pd.concat([numericalData, operatingSystems], axis=1)

numericalData.to_csv("../../ModifiedData/ModelData/NumericalData.csv", index=False)

# numericalData = pd.read_csv("../../ModifiedData/ModelData/NumericalData.csv")
# print(numericalData)