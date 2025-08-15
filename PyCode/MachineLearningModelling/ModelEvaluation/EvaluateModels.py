import matplotlib.pyplot as plt
from sklearn.model_selection import LearningCurveDisplay

from FinalModels import *
from sklearn.metrics import (precision_score, precision_recall_curve,
                             confusion_matrix, ConfusionMatrixDisplay, recall_score, accuracy_score)

# EVALUATE KNN

predictedTest = knnModel.predict(predictorTest)

print(precision_score(targetTest, predictedTest))

predictedProbs = knnModel.predict_proba(predictorTest)[:,1]
precision, recall, _ = precision_recall_curve(targetTest, predictedProbs)
plt.fill_between(precision, recall)
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.show()

# Learning curve model
# scaler = preprocessing.MinMaxScaler()
# scaler.fit(predictorData)
# predictorData = scaler.transform(predictorData)
#
# LearningCurveDisplay.from_estimator(knnModel, predictorData, targetData)
# plt.show()

predictedTest = knnModel.predict(predictorTest)
cm = confusion_matrix(targetTest, predictedTest)

positiveAcc = recall_score(targetTest, predictedTest)
negativeAcc = recall_score(targetTest, predictedTest, pos_label=0)
print(positiveAcc, negativeAcc)

cmD = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BelowMedian", "AboveMedian"])
cmD.plot()
plt.yticks(rotation=90)
plt.show()

# EVALUATE DECISION TREE

predictedTest = dtModel.predict(predictorTest)
cm = confusion_matrix(targetTest, predictedTest)

positiveAcc = recall_score(targetTest, predictedTest)
negativeAcc = recall_score(targetTest, predictedTest, pos_label=0)
print(positiveAcc, negativeAcc)

cmD = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BelowMedian", "AboveMedian"])
cmD.plot()
plt.yticks(rotation=90)
plt.show()


# EVALUATE FOREST

predictedTest = rfModel.predict(predictorTest)
cm = confusion_matrix(targetTest, predictedTest)

positiveAcc = recall_score(targetTest, predictedTest)
negativeAcc = recall_score(targetTest, predictedTest, pos_label=0)
scoreAcc = accuracy_score(targetTest, predictedTest)
print(scoreAcc, positiveAcc, negativeAcc)

cmD = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BelowMedian", "AboveMedian"])
cmD.plot()
plt.yticks(rotation=90)
plt.show()


# EVALUATE HISTOGRAM GRADIENT BOOSTING

predictedTest = hgbModel.predict(predictorTest)
cm = confusion_matrix(targetTest, predictedTest)

positiveAcc = recall_score(targetTest, predictedTest)
negativeAcc = recall_score(targetTest, predictedTest, pos_label=0)
scoreAcc = accuracy_score(targetTest, predictedTest)
print(scoreAcc, positiveAcc, negativeAcc)

cmD = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BelowMedian", "AboveMedian"])
cmD.plot()
plt.yticks(rotation=90)
plt.show()
