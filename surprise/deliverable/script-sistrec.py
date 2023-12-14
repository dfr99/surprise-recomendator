# -*- coding: utf-8 -*-

import copy
import random
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from collections import defaultdict

from surprise.reader import Reader
from surprise.trainset import Trainset
from surprise import accuracy, Dataset, SVD, KNNWithZScore, NormalPredictor
from surprise.model_selection import (
    cross_validate,
    KFold,
    GridSearchCV,
    train_test_split,
)

### Global variables

file_path = "../data/ml-latest-small/ratings.csv"
df = pd.read_csv(file_path, delimiter=",")

user_column = "userId"
product_column = "movieId"
ratings_column = "rating"
user_min_ratings = 20
product_min_ratings = 10

### Functions definitions


def set_my_folds(dataset, nfolds=5, shuffle=True):
    folds = []
    raw_ratings = dataset.raw_ratings

    if shuffle:
        raw_ratings = random.sample(raw_ratings, len(raw_ratings))

    chunk_size = int(1 / nfolds * len(raw_ratings))
    thresholds = [chunk_size * x for x in range(0, nfolds)]

    print("set_my_folds> len(raw_ratings): %d" % len(raw_ratings))

    for th in thresholds:
        test_raw_ratings = raw_ratings[th : th + chunk_size]
        train_raw_ratings = raw_ratings[:th] + raw_ratings[th + chunk_size :]

        print(
            "set_my_folds> threshold: %d, len(train_raw_ratings): %d, len(test_raw_ratings): %d"
            % (th, len(train_raw_ratings), len(test_raw_ratings))
        )

        folds.append((train_raw_ratings, test_raw_ratings))

    return folds


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    precisions = dict()
    recalls = dict()
    user_est_true = defaultdict(list)

    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


######################################################################################################################

print("Tamaño de la matriz (filas, columnas):", df.shape)
df.info()
print("\nNúmero de valores duplicados:", df.duplicated().sum())
print("Recuento de valores únicos")
print(df.nunique())

######################################################################################################################

product_counts = df.groupby(product_column)[product_column].count()
products_to_drop = product_counts[product_counts < product_min_ratings].index
filtered_df = df[~df[product_column].isin(products_to_drop)]

print("\nRecuento de valores únicos tras filtrar productos")
print(filtered_df.nunique())
print("Tamaño de la matriz (filas, columnas):", filtered_df.shape)

user_counts = filtered_df.groupby(user_column)[user_column].count()
users_to_drop = user_counts[user_counts < user_min_ratings].index
filtered_df = filtered_df[~filtered_df[user_column].isin(users_to_drop)]

print("\nRecuento de valores únicos tras filtrar productos y usuarios")
print(filtered_df.nunique())
print("Tamaño de la matriz (filas, columnas):", filtered_df.shape)

######################################################################################################################

plt.hist(filtered_df[user_column], bins=50, color="blue", edgecolor="black")

plt.title("Número de puntuaciones por usuario")
plt.xlabel(user_column)
plt.ylabel("Número de puntuaciones")

plt.show()

plt.hist(filtered_df[product_column], bins=50, color="blue", edgecolor="black")

plt.title("Número de puntuaciones por producto")
plt.xlabel(product_column)
plt.ylabel("Número de puntuaciones")

plt.show()

plt.hist(
    filtered_df.groupby(user_column)[ratings_column].mean(),
    bins=50,
    color="orange",
    edgecolor="black",
)

plt.title("Media de puntuaciones por usuario")
plt.xlabel(ratings_column)
plt.ylabel("Media del número de puntuaciones")

plt.show()

plt.hist(
    filtered_df.groupby(product_column)[ratings_column].mean(),
    bins=50,
    color="orange",
    edgecolor="black",
)

plt.title("Media de puntuaciones por producto")
plt.xlabel(ratings_column)
plt.ylabel("Media del número de puntuaciones")

plt.show()

ratings_counts = filtered_df[ratings_column].value_counts().sort_index()
ratings_counts.plot(kind="bar", color="green", edgecolor="black")

plt.title("Distribución de las puntuaciones")
plt.xlabel(ratings_column)
plt.ylabel("Número de puntuaciones")

plt.show()

######################################################################################################################

my_seed = 144821081
random.seed(my_seed)
np.random.seed(my_seed)

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(
    filtered_df[[user_column, product_column, ratings_column]], reader
)
folds = set_my_folds(data)

######################################################################################################################

knn_param_grid = {
    "k": [25, 50, 100],
    "min_k": [1, 2, 5],
    "sim_options": {"name": ["pearson"], "user_based": "True"},
}

for i, (train_ratings, test_ratings) in enumerate(folds):
    print("Fold: %d" % i)

    knn_gs = GridSearchCV(
        KNNWithZScore, knn_param_grid, measures=["MAE"], cv=3, n_jobs=-1
    )

    # fit parameter must have a raw_ratings attribute
    train_dataset = copy.deepcopy(data)
    train_dataset.raw_ratings = train_ratings
    knn_gs.fit(train_dataset)

    # best MAE score
    print(
        "Grid search>\nmae=%.3f, cfg=%s"
        % (knn_gs.best_score["mae"], knn_gs.best_params["mae"])
    )

    # We can now use the algorithm that yields the best MAE
    knn_algo = knn_gs.best_estimator["mae"]

    # We train the algorithm with the whole train set
    knn_algo.fit(train_dataset.build_full_trainset())

    # test parameter must be a test_set
    test_dataset = copy.deepcopy(data)
    test_dataset.raw_ratings = test_ratings
    test_set = test_dataset.construct_testset(raw_testset=test_ratings)

    svd_predictions = knn_algo.test(test_set)

    # Compute and print MAE
    print("Test>")
    accuracy.mae(svd_predictions, verbose=True)

######################################################################################################################

normal_predictor = cross_validate(
    NormalPredictor(), data=data, measures=["mae"], cv=3, verbose=True
)

knn_zscore = cross_validate(
    KNNWithZScore(k=50, min_k=5, verbose=False),
    data=data,
    measures=["mae"],
    cv=3,
    verbose=True,
)

svd = cross_validate(SVD(n_factors=25), data=data, measures=["mae"], cv=3, verbose=True)

######################################################################################################################

sizes = [1, 2, 5, 10]
algos = [NormalPredictor, KNNWithZScore, SVD]
algos_names = ["Normal Predictor", "KNNZScore", "SVD"]
algos_parameters = [
    {},
    {
        "k": [50],
        "min_k": [5],
        "sim_options": {"name": ["pearson"], "user_based": "True"},
    },
    {"n_factors": [25]},
]

precision_recall_array = [[] for _ in range(len(algos))]

# For each algorithm
for algo_pos in range(len(algos)):
    # For each size
    for size in sizes:
        # Temporal array to store precision and recall for each fold
        tmp = np.array((0, 0))
        # Perform the model training and testing
        for i, (train_ratings, test_ratings) in enumerate(folds):
            print("Fold: %d" % i)
            grid_search = GridSearchCV(
                algos[algo_pos],
                algos_parameters[algo_pos],
                measures=["mae"],
                cv=3,
                n_jobs=-1,
            )

            train_dataset = copy.deepcopy(data)
            train_dataset.raw_ratings = train_ratings
            grid_search.fit(train_dataset)
            algo = grid_search.best_estimator["mae"]
            algo.fit(train_dataset.build_full_trainset())

            test_dataset = copy.deepcopy(data)
            test_dataset.raw_ratings = test_ratings
            test_set = test_dataset.construct_testset(raw_testset=test_ratings)
            predictions = algo.test(test_set)
            
            precision, recall = precision_recall_at_k(predictions, size, threshold=4)
            tmp = np.add(
                tmp,
                (
                    np.array(
                        (
                            (
                                sum(precision for precision in precision.values())
                                / len(precision)
                            )
                            / 5,
                            (sum(recall for recall in recall.values()) / len(recall))
                            / 5,
                        )
                    )
                ),
            )

        precision_recall_array[algo_pos].append(tmp)


# Plot precision and recall
canvas = plt.figure()
figure = canvas.add_subplot(1, 1, 1)

figure.set_title("Precision recall at K")
figure.set_xlabel("Recall")
figure.set_ylabel("Precision")

for i in range(len(algos)):
    # item[0] -> precision, y axis
    # item[1] -> recall, x axis
    plt.plot(
        [item[1] for item in precision_recall_array[i]],
        [item[0] for item in precision_recall_array[i]],
        label=algos_names[i],
        marker="x",
    )

plt.legend(loc="best")
plt.show()
