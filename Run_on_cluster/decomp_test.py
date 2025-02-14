import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from VersaQT.data_manip import *
from symbolic_reconstruct import *

from pysr import PySRRegressor
import sympy as smp


def time_train_test_split(X, y, train_size=0.8):
    X_ = X.reshape(-1, )
    y_ = y.reshape(-1, )

    size = X_.size
    div_id = int(X_.size*train_size)

    X_train, X_test = X_[:div_id], X_[div_id:]
    y_train, y_test = y_[:div_id], y_[div_id:]

    return X_train, y_train, X_test, y_test



if __name__ == "__main__":
    # corn = yf.download("ZC=F", start="2020-01-01", end="2023-12-31", interval="1d")
    # corn.columns = corn.columns.get_level_values(0)

    # corn.reset_index(inplace=True)
    with open("corn-data.pkl", "rb") as file:
        corn = pickle.load(file)


    X = np.c_[corn.index.values]
    y = np.c_[corn.Close.values]

    pysr = PySRRegressor(
        binary_operators=["+", "-", "*", "/"], 
        unary_operators=["exp", "cos", "sin", "tan"],
        temp_equation_file=True,
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        niterations=60,
        populations=20,
        population_size=60,
        maxsize=40,
        warm_start=False,
        verbosity=False,
        progress=False
    )

    train_sizes = [0.95, 0.9, 0.8, 0.85, 0.6, 0.4]
    for train_size in train_sizes:
        X_train, y_train, X_test, y_test = time_train_test_split(X, y, train_size=train_size)

        SRReconstruct = SymbolicReconstruct(pysr, f"save_results/new_tests-{train_size}")
        SRReconstruct.fit(np.c_[X_train], np.c_[y_train], seed=42)
        decomp_size = SRReconstruct.decomposition.shape[0]-1
        SRReconstruct.set_use_indexes(list(range(decomp_size, -1, -1)))
        

        SRReconstruct.multiple_reconstruct(10)