import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import lines

from sklearn.cluster import KMeans
from sklearn import preprocessing
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 220)
pd.set_option('display.max_colwidth', 30)

df = pd.read_csv("../../ModifiedData/TransformedData/CleanedDataWithFrequency.csv")
sampleColumns = ["YearsCodePro", "EdLevelOrdinals", "OrgSizeOrdinals", "AgeOrdinals"]

# Fetch and split data into Above and Below median groups
ordinalData = df[["HighIncomeDev", *sampleColumns]].copy()
belowMedianData = ordinalData.loc[df["HighIncomeDev"] == 0.0][sampleColumns]
aboveMedianData = ordinalData.loc[df["HighIncomeDev"] == 1.0][sampleColumns]


# Produce Elbow Visualisations of both AM/BM groups
def scale(dataList : []):
    scaledDataList = []
    scalers = []

    for data in dataList:
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(data)
        scalers.append(scaler)
        scaledData = scaler.transform(data)
        scaledDataList.append(pd.DataFrame(scaledData, columns=sampleColumns))

    return [*scaledDataList, scalers]


belowMedianData, aboveMedianData, scalers = scale([belowMedianData, aboveMedianData])

# Elbow Visualisation to determine K clusters
visualiser = KElbowVisualizer(KMeans(), k=(2,10), timings=False)
visualiser.fit(aboveMedianData)
line = plt.gca().get_lines()[:2]
line[0].set_color('c')
line[0].set_label('Above-Median Data')
line[1].set_color('c')
visualiser = KElbowVisualizer(KMeans(), k=(2,10), timings=False)
visualiser.fit(belowMedianData)
line = plt.gca().get_lines()[2:4]
line[0].set_color('b')
line[0].set_label('Below-Median Data')
line[1].set_color('b')
plt.legend(plt.gca().get_lines(), ['Cyan', 'Blue'], loc='lower right')
visualiser.show()


# Could directly use values from visualiser.elbow_score_, but those may change
# Hardcode with selected results for better graph consistency
clusters = [4, 5]
titles = ["Clustering for Below-Median Data", "Clustering for Above-Median Data"]
titlesCentre = ["Cluster Centre values for Below-Median Data", "Cluster Centre values for Above-Median Data"]

clusterCentresDF = pd.DataFrame(columns=sampleColumns)

for i, data in enumerate([belowMedianData, aboveMedianData]):

    # Create KMeans Model, fit and produce cluster groups for data
    # km = KMeans(n_clusters=k, init='random', n_init=10, max_iter=300,tol=1e-04, random_state=0)
    km = KMeans(n_clusters=clusters[i], init="k-means++", n_init=10)
    y = km.fit_predict(data)
    data['ClusterLabel'] = y

    # Cluster Characteristics
    clusterCentres = km.cluster_centers_
    df_centre = pd.DataFrame(clusterCentres, columns=sampleColumns)
    unscaledCentres = scalers[i].inverse_transform(df_centre)
    print("ClusterCentres\n", pd.DataFrame(unscaledCentres, columns=sampleColumns))

    # Plot cluster centre points
    plt.scatter(data=df_centre, x="YearsCodePro", y="EdLevelOrdinals", marker='D', c=df_centre.index, cmap="Set1", vmin=df_centre.index.min(), vmax=df_centre.index.max())
    for c, label in enumerate(df_centre.index):
        plt.annotate(label, (df_centre["YearsCodePro"][c]-0.01, df_centre["EdLevelOrdinals"][c]-0.01))

    # Plot Cluster Data for select cluster(s)
    clusterData = data.loc[data["ClusterLabel"].isin([0, 1])]
    plt.scatter(data=clusterData, x="YearsCodePro", y="EdLevelOrdinals", c="ClusterLabel", cmap="Set1", vmin=df_centre.index.min(), vmax=df_centre.index.max(), alpha=0.5)
    plt.xlabel("Years Coding Professionally (0 to 50 in rescaled values)")
    plt.ylabel("Education Level (in scaled ordinal values)")
    plt.title(titles[i])

    # produce legend
    legendClusterCentre = lines.Line2D([], [], marker='D', color='k', linestyle='None', label="Cluster Centre")
    legendClusterData = lines.Line2D([], [], marker='o', color='k', linestyle='None', label="Cluster Data")
    plt.legend(handles=[legendClusterCentre, legendClusterData])
    plt.show()

    # Plot Cluster Characteristics
    df_centre.plot.bar()
    plt.xlabel("Cluster Group")
    plt.ylabel("Scaled Data Values for Cluster Centre")
    plt.title(titlesCentre[i])
    plt.show()

    # Plot Cluster Silhouettes
    visualiser = SilhouetteVisualizer(KMeans(n_clusters=clusters[i], init="k-means++", n_init=10))
    visualiser.fit(data)
    visualiser.show()
    print(visualiser.silhouette_score_)


    if i > 0:
        df_centre.index += clusters[i-1]
    clusterCentresDF = pd.concat([clusterCentresDF, df_centre], axis=0)

aboveMedianData["ClusterLabel"] += clusters[0]
aboveMedianData["HighIncomeDev"] = True
belowMedianData["HighIncomeDev"] = False
compiledDF = pd.concat([belowMedianData, aboveMedianData], axis=0, ignore_index=True)
print(compiledDF)

clusterCentresDF.columns = ["ClCentre_" + col for col in clusterCentresDF.columns]
print(clusterCentresDF)

compiledDF = pd.merge(left=compiledDF, right=clusterCentresDF, right_index=True, left_on="ClusterLabel")
compiledDF.to_csv("../../ModifiedData/ModelData/NumericalClustered.csv")