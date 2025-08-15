from matplotlib import pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve, auc

from FinalModels import models, ensembleModels, predictorTest, targetTest

# Produce ROC curve for each model

for model in models:
    predictionScores = model.predict_proba(predictorTest)[:, 1]
    falsePosRate, truePosRate, _ = roc_curve(targetTest, predictionScores)

    # Area under curve
    rocAuc = auc(falsePosRate, truePosRate)

    # Model name
    modelName = str(model).split(sep='(')[0]

    # plot curve
    plt.plot(falsePosRate, truePosRate, label=f"{modelName} auc: {rocAuc}")

# plot random guessing as 50% line
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# Second ROC Curve using ensemble models

for model in [*models, *ensembleModels]:
    predictionScores = model.predict_proba(predictorTest)[:, 1]
    falsePosRate, truePosRate, _ = roc_curve(targetTest, predictionScores)

    # Area under curve
    rocAuc = auc(falsePosRate, truePosRate)

    # Model name
    modelName = str(model).split(sep='(')[0]

    # plot curve
    plt.plot(falsePosRate, truePosRate, label=f"{modelName} auc: {rocAuc}")

# plot random guessing as 50% line
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
