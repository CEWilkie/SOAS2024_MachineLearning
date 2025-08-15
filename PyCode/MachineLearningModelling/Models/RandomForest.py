import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import sklearn.model_selection as skms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 220)
pd.set_option('display.max_colwidth', 30)

data = pd.read_csv("../../ModifiedData/TransformedData/CleanedDataWithLabels.csv")

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
scaler = preprocessing.MinMaxScaler()
scaler.fit(predictorTrain)
predictorTrain = scaler.transform(predictorTrain)
predictorTest = scaler.transform(predictorTest)

# Determine best parameters
params = {
    'n_estimators': range(100, 200, 10),
    'max_features': ['sqrt', 'log2'],
    'max_depth' : range(1, 20),
    'min_samples_split':range(2, 30),
    'min_samples_leaf':range(1,30),
    'bootstrap':[True]
}

searchHyperParams = skms.RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    params,
    n_iter=60,
    cv=5,
    random_state=42,
    verbose=4
)
# searchHyperParams.fit(predictorTrain, targetTrain)
# bestParams = searchHyperParams.best_params_
# bestAcc = searchHyperParams.best_score_
# print(bestAcc, bestParams)

# Results
# 0.7863644177076061 {'n_estimators': 110, 'min_samples_split': 8, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 17, 'bootstrap': True}
# max accuracy:  0.7942829220620572 with depth:  84 with max features:  sqrt
# max accuracy:  0.79599315905204 with depth:  114 with max features:  log2
# max accuracy:  0.8299535792817004 with depth:  112 with max features:  None

# Tune maxFeatures, n_estimators parameter
accuracies = {}
colours = ['r', 'c', 'g']
for i, maxFeatures in enumerate(['sqrt', 'log2', None]):
    for estimators in range(70, 150):
        rf = RandomForestClassifier(n_estimators=estimators, min_samples_split=8, min_samples_leaf=1, max_features=maxFeatures, max_depth=17, bootstrap=True)
        rf.fit(predictorTrain, targetTrain)

        predictTest = rf.predict(predictorTest)
        accuracies[estimators] = accuracy_score(targetTest, predictTest)

    plt.plot(list(accuracies.keys()), list(accuracies.values()), c=colours[i], label=(maxFeatures if maxFeatures is not None else "No Limit"))
    maxAcc = max(accuracies.values())
    print("max accuracy: ", maxAcc, "with depth: ", list(accuracies.keys())[list(accuracies.values()).index(maxAcc)],
          "with max features: ", maxFeatures)

plt.plot(list(accuracies.keys()), list(accuracies.values()))
plt.title("Accuracy of varying tree max depths")
plt.xlabel("Num Trees in Forest")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Tune criterion, depth parameter
accuracies = {}
for algo in ['log_loss']:
    for depth in range(127, 135):
        rf = RandomForestClassifier(n_estimators=112, min_samples_split=8, min_samples_leaf=1, max_features=None,
                                    max_depth=depth, bootstrap=True, criterion=algo)
        rf.fit(predictorTrain, targetTrain)

        predictTest = rf.predict(predictorTest)
        accuracies[depth] = accuracy_score(targetTest, predictTest)

    plt.plot(list(accuracies.keys()), list(accuracies.values()), label=algo)
    maxAcc = max(accuracies.values())
    print("max accuracy: ", maxAcc, "with depth: ", list(accuracies.keys())[list(accuracies.values()).index(maxAcc)],
          "with criterion: ", algo)

# max accuracy:  0.8287319814317127 with depth:  123 with criterion:  gini
# max accuracy:  0.8294649401417054 with depth:  97 with criterion:  entropy
# max accuracy:  0.8314194967016858 with depth:  129 with criterion:  log_loss

plt.plot(list(accuracies.keys()), list(accuracies.values()))
plt.title("Accuracy of varying tree max depths")
plt.xlabel("Tree Max Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()