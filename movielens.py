import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"  # Set local IP to avoid hostname warnings
import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)

import sys
import numpy as np
import pandas as pd
import cornac
 
try:
    import pyspark
    from recommenders.utils.spark_utils import start_or_get_spark
except ImportError:
    pass  # skip this import if we are not in a Spark environment

try:
    import tensorflow as tf # NOTE: TF needs to be imported before PyTorch, otherwise we get an error
    tf.get_logger().setLevel("ERROR") # only show error messages
    import torch
    from recommenders.utils.gpu_utils import get_cuda_version, get_cudnn_version
except ImportError:
    pass  # skip this import if we are not in a GPU environment

try:
    import surprise # Put SVD surprise back in core deps when #2224 is fixed
except:
    pass

current_path = "/content/recommenders/examples/06_benchmarks"
# current_path = os.path.join(os.getcwd(), "examples", "06_benchmarks")

#  # To execute the notebook programmatically from root folder
sys.path.append(current_path)
from benchmark_utils import *
from recommenders.datasets import movielens
from recommenders.utils.general_utils import get_number_processors
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.utils.notebook_utils import store_metadata


print(f"System version: {sys.version}")
print(f"Number of cores: {get_number_processors()}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Cornac version: {cornac.__version__}")

try:
    print(f"Surprise version: {surprise.__version__}") # Put SVD surprise back in core deps when #2224 is fixed
except NameError:
    pass

try:
    print(f"PySpark version: {pyspark.__version__}")
except NameError:
    pass  # skip this import if we are not in a Spark environment

try:
    print(f"CUDA version: {get_cuda_version()}")
    print(f"CuDNN version: {get_cudnn_version()}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"PyTorch version: {torch.__version__}")
except NameError:
    pass  # skip this import if we are not in a GPU environment

# %load_ext autoreload
# %autoreload 2

try:
    spark = start_or_get_spark("PySpark", memory="10g")
    spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")
    # Suppress Spark warnings
    spark.sparkContext.setLogLevel("ERROR")
    log4j = spark._jvm.org.apache.log4j
    log4j.LogManager.getLogger("org").setLevel(log4j.Level.ERROR)
    log4j.LogManager.getLogger("akka").setLevel(log4j.Level.ERROR)
    log4j.LogManager.getLogger("org.apache.spark").setLevel(log4j.Level.ERROR)
    log4j.LogManager.getLogger("org.spark_project").setLevel(log4j.Level.ERROR)
except NameError:
    pass  # skip this import if we are not in a Spark environment

# Fix random seeds to make sure out runs are reproducible
np.random.seed(SEED)
try:
    tf.random.set_seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
except NameError:
    pass  # skip this import if we are not in a GPU environment

"""## Parameters"""

data_sizes = ["100k"] # Movielens data size: 100k, 1m, 10m, or 20m
# algorithms = ["als", "svd", "sar", "ncf", "embdotbias", "bpr", "bivae", "lightgcn"]


algorithms = ["als", "svd", "sar", "ncf", "bpr"]


environments = {
    "als": "pyspark",
    "sar": "python_cpu",
    "svd": "python_cpu",
    "embdotbias": "python_gpu",
    "ncf": "python_gpu",
    "bpr": "python_cpu",
    "bivae": "python_gpu",
    "lightgcn": "python_gpu",
}

metrics = {
    "als": ["rating", "ranking"],
    "sar": ["ranking"],
    "svd": ["rating", "ranking"],
    "embdotbias": ["rating", "ranking"],
    "ncf": ["ranking"],
    "bpr": ["ranking"],
    "bivae": ["ranking"],
    "lightgcn": ["ranking"]
}

"""Algorithm parameters"""

als_params = {
    "rank": 10,
    "maxIter": 20,
    "implicitPrefs": False,
    "alpha": 0.1,
    "regParam": 0.05,
    "coldStartStrategy": "drop",
    "nonnegative": False,
    "userCol": DEFAULT_USER_COL,
    "itemCol": DEFAULT_ITEM_COL,
    "ratingCol": DEFAULT_RATING_COL,
}

sar_params = {
    "similarity_type": "jaccard",
    "time_decay_coefficient": 30,
    "time_now": None,
    "timedecay_formula": True,
    "col_user": DEFAULT_USER_COL,
    "col_item": DEFAULT_ITEM_COL,
    "col_rating": DEFAULT_RATING_COL,
    "col_timestamp": DEFAULT_TIMESTAMP_COL,
}

svd_params = {
    "n_factors": 150,
    "n_epochs": 15,
    "lr_all": 0.005,
    "reg_all": 0.02,
    "random_state": SEED,
    "verbose": False
}

embdotbias_params = {
    "n_factors": 40,
    "y_range": [0,5.5],
    "wd": 1e-1,
    "lr_max": 5e-3,
    "epochs": 15
}

ncf_params = {
    "model_type": "NeuMF",
    "n_factors": 4,
    "layer_sizes": [16, 8, 4],
    "n_epochs": 15,
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "verbose": 10
}

bpr_params = {
    "k": 200,
    "max_iter": 200,
    "learning_rate": 0.01,
    "lambda_reg": 1e-3,
    "seed": SEED,
    "verbose": False
}

bivae_params = {
    "k": 100,
    "encoder_structure": [200],
    "act_fn": "tanh",
    "likelihood": "pois",
    "n_epochs": 500,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "seed": SEED,
    "use_gpu": True,
    "verbose": False
}

lightgcn_param = {
    "model_type": "lightgcn",
    "n_layers": 3,
    "batch_size": 1024,
    "embed_size": 64,
    "decay": 0.0001,
    "epochs": 20,
    "learning_rate": 0.005,
    "eval_epoch": 5,
    "top_k": DEFAULT_K,
    "metrics": ["recall", "ndcg", "precision", "map"],
    "save_model":False,
    "MODEL_DIR":".",
}

params = {
    "als": als_params,
    "sar": sar_params,
    "svd": svd_params,
    "embdotbias": embdotbias_params,
    "ncf": ncf_params,
    "bpr": bpr_params,
    "bivae": bivae_params,
    "lightgcn": lightgcn_param,
}

prepare_training_data = {
    "als": prepare_training_als,
    "sar": prepare_training_sar,
    "svd": prepare_training_svd,
    "embdotbias": prepare_training_embdotbias,
    "ncf": prepare_training_ncf,
    "bpr": prepare_training_cornac,
    "bivae": prepare_training_cornac,
    "lightgcn": prepare_training_lightgcn,
}

prepare_metrics_data = {
    "als": lambda train, test: prepare_metrics_als(train, test),
    "embdotbias": lambda train, test: prepare_metrics_embdotbias(train, test),
}

trainer = {
    "als": lambda params, data: train_als(params, data),
    "svd": lambda params, data: train_svd(params, data),
    "sar": lambda params, data: train_sar(params, data),
    "embdotbias": lambda params, data: train_embdotbias(params, data),
    "ncf": lambda params, data: train_ncf(params, data),
    "bpr": lambda params, data: train_bpr(params, data),
    "bivae": lambda params, data: train_bivae(params, data),
    "lightgcn": lambda params, data: train_lightgcn(params, data),
}

rating_predictor = {
    "als": lambda model, test: predict_als(model, test),
    "svd": lambda model, test: predict_svd(model, test),
    "embdotbias": lambda model, test: predict_embdotbias(model, test),
}

ranking_predictor = {
    "als": lambda model, test, train: recommend_k_als(model, test, train),
    "sar": lambda model, test, train: recommend_k_sar(model, test, train),
    "svd": lambda model, test, train: recommend_k_svd(model, test, train),
    "embdotbias": lambda model, test, train: recommend_k_embdotbias(model, test, train),
    "ncf": lambda model, test, train: recommend_k_ncf(model, test, train),
    "bpr": lambda model, test, train: recommend_k_bpr(model, test, train),
    "bivae": lambda model, test, train: recommend_k_bivae(model, test, train),
    "lightgcn": lambda model, test, train: recommend_k_lightgcn(model, test, train),
}

rating_evaluator = {
    "als": lambda test, predictions: rating_metrics_pyspark(test, predictions),
    "svd": lambda test, predictions: rating_metrics_python(test, predictions),
    "embdotbias": lambda test, predictions: rating_metrics_python(test, predictions)
}


ranking_evaluator = {
    "als": lambda test, predictions, k: ranking_metrics_pyspark(test, predictions, k),
    "sar": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "svd": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "embdotbias": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "ncf": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "bpr": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "bivae": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "lightgcn": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
}

def generate_summary(data, algo, k, train_time, time_rating, rating_metrics, time_ranking, ranking_metrics):
    summary = {"Data": data, "Algo": algo, "K": k, "Train time (s)": train_time, "Predicting time (s)": time_rating, "Recommending time (s)": time_ranking}
    if rating_metrics is None:
        rating_metrics = {
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
            "Explained Variance": np.nan,
        }
    if ranking_metrics is None:
        ranking_metrics = {
            "MAP": np.nan,
            "nDCG@k": np.nan,
            "Precision@k": np.nan,
            "Recall@k": np.nan,
        }
    summary.update(rating_metrics)
    summary.update(ranking_metrics)
    return summary

"""## Benchmark loop"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# For each data size and each algorithm, a recommender is evaluated.
cols = ["Data", "Algo", "K", "Train time (s)", "Predicting time (s)", "RMSE", "MAE", "R2", "Explained Variance", "Recommending time (s)", "MAP", "nDCG@k", "Precision@k", "Recall@k"]
df_results = pd.DataFrame(columns=cols)

for data_size in data_sizes:
    # Load the dataset
    df = movielens.load_pandas_df(
        size=data_size,
        header=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL]
    )
    print("Size of Movielens {}: {}".format(data_size, df.shape))

    # Split the dataset
    df_train, df_test = python_stratified_split(df,
                                                ratio=0.75,
                                                min_rating=1,
                                                filter_by="item",
                                                col_user=DEFAULT_USER_COL,
                                                col_item=DEFAULT_ITEM_COL
                                                )

    # Loop through the algos
    for algo in algorithms:
        print(f"\nComputing {algo} algorithm on Movielens {data_size}")

        # Data prep for training set
        train = prepare_training_data.get(algo, lambda x,y:(x,y))(df_train, df_test)
# 
        # Get model parameters
        model_params = params[algo]

#         # Train the model
        model, time_train = trainer[algo](model_params, train)
        print(f"Training time: {time_train}s")
# 
#         # Predict and evaluate
        train, test = prepare_metrics_data.get(algo, lambda x,y:(x,y))(df_train, df_test)
# 
        if "rating" in metrics[algo]:
            # Predict for rating
            preds, time_rating = rating_predictor[algo](model, test)
            print(f"Rating prediction time: {time_rating}s")
# 
#             # Evaluate for rating
            ratings = rating_evaluator[algo](test, preds)
        else:
            ratings = None
            time_rating = np.nan

        if "ranking" in metrics[algo]:
            # Predict for ranking
            top_k_scores, time_ranking = ranking_predictor[algo](model, test, train)
            print(f"Ranking prediction time: {time_ranking}s")
# 
#             # Evaluate for rating
            rankings = ranking_evaluator[algo](test, top_k_scores, DEFAULT_K)
        else:
            rankings = None
            time_ranking = np.nan
# 
#         # Record results
        summary = generate_summary(data_size, algo, DEFAULT_K, time_train, time_rating, ratings, time_ranking, rankings)
        df_results.loc[df_results.shape[0] + 1] = summary
# 
print("\nComputation finished")
#

"""## Results"""

print(df_results)
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'result_{timestamp}.csv'

df_results.to_csv(filename, index=False)

# Record results for tests - ignore this cell
for algo in algorithms:
    store_metadata(algo, df_results.loc[df_results["Algo"] == algo, "nDCG@k"].values[0])