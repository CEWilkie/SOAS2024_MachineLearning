import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import sklearn.model_selection as skms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 220)
pd.set_option('display.max_colwidth', 30)

# Obtain predictor and target data
data = pd.read_csv("../../ModifiedData/TransformedData/CleanedDataWithFrequency.csv")
target = "HighIncomeDev"
columns = data.columns.drop(target)
# columns = ["AgeOrdinals", "EdLevelOrdinals", "YearsCodePro", "OrgSizeOrdinals", "Country", "DevType", "Industry"]

predictorData = data[columns]
targetData = data[target]

# Split into train and test data
predictorTrain, predictorTest, targetTrain, targetTest = (
    skms.train_test_split(predictorData, targetData, test_size=0.2, random_state=7))

# Scale data to avoid weighting
# This model actually works best without scaling?
scaler = preprocessing.MinMaxScaler()
scaler.fit(predictorTrain)
predictorTrain = scaler.transform(predictorTrain)
predictorTest = scaler.transform(predictorTest)


# Create baseline model
baselineKNC = KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="manhattan")
baselineKNC.fit(predictorTrain, targetTrain)

# Attempt to find best hyperparameters
params = {
    'n_neighbors' : range(5, 100),
    'weights' : ['uniform', 'distance'],
    'metric' : ['manhattan', 'euclidean'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    'leaf_size': range(10, 51, 3),
    'p': [1, 2]
}

searchHyperParams = skms.RandomizedSearchCV(
    KNeighborsClassifier(),
    params,
    n_iter=60,
    cv=30,
    n_jobs=1,
    verbose=4
)

# use best params to create new model
# searchHyperParams.fit(predictorTrain, targetTrain)
# bestParams = searchHyperParams.best_params_
# bestAcc = searchHyperParams.best_score_
# print(bestAcc, bestParams)

# results
# Accuracy 0.7493423396175688, originally 91 k
bestParams = {'weights': 'distance', 'p': 1, 'n_neighbors': 143, 'metric': 'manhattan', 'leaf_size': 46, 'algorithm': 'kd_tree'}
k = bestParams['n_neighbors']

# Test Uniform and Distance weighting
testResults = {}
testResultsDistance = {}
highestAccUni = {'k':0, 'v':0}
highestAccDist = {'k':0, 'v':0}
for newK in range(k-10, k+10, 1):
    print(newK)
    knnModel = KNeighborsClassifier(weights='uniform', p=1, n_neighbors=newK, metric='manhattan', leaf_size=46, algorithm='kd_tree')
    knnModel.fit(predictorTrain, targetTrain)

    targetPredict = knnModel.predict(predictorTest)
    testResults[newK] = accuracy_score(targetTest, targetPredict)
    if highestAccUni['v'] < testResults[newK]:
        highestAccUni['k'] = newK
        highestAccUni['v'] = testResults[newK]

    # Model using Weighted Distance
    knnModel = KNeighborsClassifier(weights='distance', p=1, n_neighbors=newK, metric='manhattan', leaf_size=46, algorithm='kd_tree')
    knnModel.fit(predictorTrain, targetTrain)

    targetPredict = knnModel.predict(predictorTest)
    testResultsDistance[newK] = accuracy_score(targetTest, targetPredict)
    if highestAccDist['v'] < testResultsDistance[newK]:
        highestAccDist['k'] = newK
        highestAccDist['v'] = testResultsDistance[newK]


print(f"uniform best is acc {highestAccUni['v']}, with k {highestAccUni['k']}")
print(f"distance best is acc {highestAccDist['v']}, with k {highestAccDist['k']}")
plt.plot(list(testResults.keys()), list(testResults.values()), c='b', label="Uniform Weighting")
plt.plot(list(testResultsDistance.keys()), list(testResultsDistance.values()), c='c', label="Distance Weighting")
plt.title("Accuracy of Uniform and Distance Weighting")
plt.xlabel("K Neighbors")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Test alternative algorithms
highestAcc = {'k':0, 'v':0, 'a':''}
for algo in ['brute', 'kd_tree', 'ball_tree']:
    highestAlgoAcc = {'k':0, 'v':0}
    for newK in range(k-20, k+20):
        # Model using Uniform Weight
        knnModel = KNeighborsClassifier(weights='distance', p=1, n_neighbors=newK, metric='manhattan', leaf_size=46, algorithm=algo)
        knnModel = knnModel.fit(predictorTrain, targetTrain)

        targetPredict = knnModel.predict(predictorTest)
        testResults[newK] = accuracy_score(targetTest, targetPredict)
        if highestAlgoAcc['v'] < testResults[newK]:
            highestAlgoAcc['k'] = newK
            highestAlgoAcc['v'] = testResults[newK]

        # Model using Weighted Distance
        # knnModel = KNeighborsClassifier(weights='uniform', p=1, n_neighbors=newK, metric='manhattan', leaf_size=43, algorithm='auto')
        # knnModel.fit(predictorTrain, targetTrain)
        #
        # targetPredict = knnModel.predict(predictorTest)
        # testResultsDistance[newK] = accuracy_score(targetTest, targetPredict)
    plt.plot(list(testResults.keys()), list(testResults.values()), label=algo)
    print(f"best of {algo} is acc {highestAlgoAcc['v']}, with k {highestAlgoAcc['k']}")
    if highestAcc['v'] < highestAlgoAcc['v']:
        highestAcc['k'] = highestAlgoAcc['k']
        highestAcc['v'] = highestAlgoAcc['v']
        highestAcc['a'] = algo


print(f"best is acc {highestAcc['v']}, with k {highestAcc['k']}, algo {highestAcc['a']}")
plt.title("Accuracy of alternative KNN Algorithms")
plt.xlabel("n_neighbors hyperparameter")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
