import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, recall_score, accuracy_score)

from FinalModels import *
from sklearn.ensemble import VotingClassifier, BaggingClassifier, StackingClassifier

# Create new Voting Classifier from prior models
allModels = [("knn", knnModel), ("dt", dtModel), ("rf", rfModel)]
accuracies = {}

vcModel = VotingClassifier(estimators=allModels)
vcModel.fit(predictorTrain, targetTrain)

predicted = vcModel.predict(predictorTest)
accuracies["Original"] = accuracy_score(targetTest, predicted)

# Determine affect of removing a particular model
for model in allModels:
    # Create ensemble from all models
    estimators = allModels.copy()
    estimators.remove(model)
    vc = VotingClassifier(estimators=estimators)
    vc.fit(predictorTrain, targetTrain)

    predicted = vc.predict(predictorTest)

    # models have difficulties predicting for high-income,

    accuracies[model[0]] = accuracy_score(targetTest, predicted)
    print(accuracies[model[0]])

plt.bar(range(len(accuracies)), list(accuracies.values()))
plt.xticks(range(len(accuracies)), list(accuracies.keys()))
plt.xlabel("Influence of removing a model upon the final accuracy")
plt.ylabel("Accuracy")
plt.show()

# Evaluate the 3-model Voting Classifier
predictedTest = vcModel.predict(predictorTest)
cm = confusion_matrix(targetTest, predictedTest)

positiveAcc = recall_score(targetTest, predictedTest)
negativeAcc = recall_score(targetTest, predictedTest, pos_label=0)
print(positiveAcc, negativeAcc)

cmD = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BelowMedian", "AboveMedian"])
cmD.plot()
plt.yticks(rotation=90)
plt.show()


# Bagging Classifier
sModel = StackingClassifier([("knn", knnModel), ("dt", dtModel), ("rf", rfModel)], LogisticRegression())
sModel.fit(predictorTrain, targetTrain)

predicted = sModel.predict(predictorTest)
print(accuracy_score(targetTest, predicted))