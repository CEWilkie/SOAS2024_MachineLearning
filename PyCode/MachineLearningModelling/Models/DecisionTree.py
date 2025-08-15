import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection as skms
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

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
    'min_samples_split': range(5, 30),
    'min_samples_leaf': range(5, 30),
    'max_depth' : range(1, 30, 1),
    'criterion':['gini', 'entropy', 'log_loss']
}

searchHyperParams = skms.RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    params,
    n_iter=60,
    cv=5,
    random_state=42
)
# searchHyperParams.fit(predictorTrain, targetTrain)
# bestParams = searchHyperParams.best_params_
# bestAcc = searchHyperParams.best_score_
# print(bestAcc, bestParams)
# 0.8040811690996158 {'min_samples_split': 25, 'min_samples_leaf': 11, 'max_depth': 26, 'criterion': 'gini'}
bestValues = {'min_samples_split': 25, 'min_samples_leaf': 11, 'max_depth': 26, 'criterion': 'gini'}

bestAcc = {'c':'', 'md':0, 'a':0}
for criterion in ['gini', 'entropy', 'log_loss']:
    accuracies = {}
    for md in range(1, 30):
        clf = DecisionTreeClassifier(min_samples_split=25, min_samples_leaf=11, max_depth=md, criterion=criterion, splitter='best', random_state=32)
        clf = clf.fit(predictorTrain, targetTrain)

        predictTargets = clf.predict(predictorTest)
        accuracies[md] = accuracy_score(targetTest, predictTargets)
        if accuracies[md] > bestAcc['a']:
            bestAcc['md'] = md
            bestAcc['a'] = accuracies[md]
            bestAcc['c'] = criterion

    plt.plot(list(accuracies.keys()), list(accuracies.values()), label=criterion)
    maxAcc = max(accuracies.values())
    print("max accuracy: ", maxAcc, "with depth: ", list(accuracies.keys())[list(accuracies.values()).index(maxAcc)],
          "with criterion: ", criterion)

print(bestAcc)

plt.title("Accuracy of alternative split Criterions")
plt.xlabel("Max depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()