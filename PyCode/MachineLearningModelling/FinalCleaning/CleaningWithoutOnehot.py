import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


# ORDINAL DATA
# EdLevel, Age, Remote, ConvertedCompYearly, YearsCode, YearsCodePro, DevType, OrgSize

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 220)
pd.set_option('display.max_colwidth', 30)

df = pd.read_csv("../../ModifiedData/TransformedData/ApplyingNominals.csv")

# Encode DevType
le = LabelEncoder()
df["DevType"] = le.fit_transform(df["DevType"])

# Encode Operating Systems
operatingSystems = df["WorkOperatingSystem"].str.get_dummies(sep=';')
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

# Finalise data subset and store
df = df[["HighIncomeDev", "AgeOrdinals", "EdLevelOrdinals", "OrgSizeOrdinals", "RemoteOrdinals", "YearsCodePro", "DevType"]]
df = pd.concat([df, operatingSystems], axis=1)

df.to_csv("../../ModifiedData/ModelData/NumericalDataTree.csv", index=False)
