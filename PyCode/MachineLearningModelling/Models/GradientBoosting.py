import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn.model_selection as skms
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 220)
pd.set_option('display.max_colwidth', 30)

data = pd.read_csv("../../ModifiedData/TransformedData/CleanedDataWithFrequency.csv")

# split data
target = "HighIncomeDev"
columns = data.columns.drop(target)
# columns = ["AgeOrdinals", "EdLevelOrdinals", "YearsCodePro", "OrgSizeOrdinals", "RemoteOrdinals"]
predictorData = data[columns]
targetData = data[target]

# Split into train and test data
predictorTrain, predictorTest, targetTrain, targetTest = (
    skms.train_test_split(predictorData, targetData, test_size=0.2, random_state=7))

# Scale data to avoid weighting
# scaler = preprocessing.MinMaxScaler()
# scaler.fit(predictorTrain)
# predictorTrain = scaler.transform(predictorTrain)
# predictorTest = scaler.transform(predictorTest)

params = {
    'n_estimators': range(50, 401, 50),
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 1.0],
}

searchHyperParams = skms.RandomizedSearchCV(
    GradientBoostingClassifier(),
    params,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    verbose=4
)
# searchHyperParams.fit(predictorTrain, targetTrain)
# bestParams = searchHyperParams.best_params_
# bestAcc = searchHyperParams.best_score_
# print(bestAcc, bestParams)

# results
# 0.836337326874834
bestParams = {'subsample': 0.8, 'n_estimators': 400, 'min_samples_split': 10, 'max_depth': 3, 'learning_rate': 0.1}

# gb = GradientBoostingClassifier(subsample=0.8, n_estimators=400, min_samples_split=10, max_depth=3, learning_rate=0.1)
# gb.fit(predictorTrain, targetTrain)
#
# predictTest = gb.predict(predictorTest)
# print(accuracy_score(targetTest, predictTest))

# Determine suitable number of iterations
hgb = HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.1, random_state=42, early_stopping=True)
hgb.fit(predictorTrain, targetTrain)

predictTest = hgb.predict(predictorTest)
print(accuracy_score(targetTest, predictTest))
print(hgb.n_iter_)
plt.plot(-hgb.validation_score_)
plt.xlabel("Number of Iterations")
plt.ylabel("Root Mean Squared Error")
plt.show()

# Compare alternative max depths
accuracies = {}
for depth in range(1, 25):
    hgb = HistGradientBoostingClassifier(max_iter=1000, max_depth=depth, learning_rate=0.1, random_state=42, early_stopping=True)
    hgb.fit(predictorTrain, targetTrain)

    predictTest = hgb.predict(predictorTest)
    accuracies[depth] = accuracy_score(targetTest, predictTest)
    print(f"accuracy {accuracies[depth]} at depth {depth}")

plt.plot(list(accuracies.keys()), list(accuracies.values()))
plt.ylabel("Accuracy")
plt.xlabel("Max Depth")
plt.show()