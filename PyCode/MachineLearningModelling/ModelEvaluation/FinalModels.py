import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier, \
    VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("../../ModifiedData/TransformedData/CleanedDataWithFrequency.csv")

# split data
target = "HighIncomeDev"
columns = data.columns.drop(target)
# columns = ["AgeOrdinals", "EdLevelOrdinals", "YearsCodePro", "OrgSizeOrdinals", "RemoteOrdinals"]
predictorData = data[columns]
targetData = data[target]

# Split into train and test data
predictorTrain, predictorTest, targetTrain, targetTest = (
    train_test_split(predictorData, targetData, test_size=0.2, random_state=7))

# Scale data to avoid weighting
scaler = preprocessing.MinMaxScaler()
scaler.fit(predictorTrain)
predictorTrain = scaler.transform(predictorTrain)
predictorTest = scaler.transform(predictorTest)

# Best Initial Models

# Best Decision Tree Model
dtModel = DecisionTreeClassifier(min_samples_split=25, min_samples_leaf=11, max_depth=19, criterion='gini')
dtModel = dtModel.fit(predictorTrain, targetTrain)

# Best KNN Model
knnModel = KNeighborsClassifier(weights='distance', p=1, n_neighbors=143, metric='manhattan', leaf_size=46, algorithm='kd_tree')
knnModel = knnModel.fit(predictorTrain, targetTrain)

# Best Random Forest Model
rfModel = RandomForestClassifier(n_estimators=112, min_samples_split=8, min_samples_leaf=1, max_features=None, max_depth=129, criterion='log_loss', bootstrap=True)
rfModel.fit(predictorTrain, targetTrain)

models = [dtModel, knnModel, rfModel]

# Ensemble Models

# Voting Classifier
allModels = [("knn", knnModel), ("dt", dtModel), ("rf", rfModel)]
vcModel = VotingClassifier(estimators=allModels, voting='soft')
vcModel.fit(predictorTrain, targetTrain)

# Best Histogram Gradient Boosting Model
hgbModel = HistGradientBoostingClassifier(max_iter=1000, max_depth=8, learning_rate=0.1, random_state=42, early_stopping=True)
hgbModel.fit(predictorTrain, targetTrain)

ensembleModels = [vcModel, hgbModel]