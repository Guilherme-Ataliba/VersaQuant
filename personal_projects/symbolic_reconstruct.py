import pandas as pd
import numpy as np
import sympy as smp
from PyEMD import CEEMDAN
import pickle

def __ceemdan_decompose(series):
    series = series.reshape(-1, )
    ceemdan = CEEMDAN()  # Criando o objeto CEEMDAN
    imfs = ceemdan(series)  # Decompondo a série
    return imfs

def symbolic_reconstruct(model, X, y, y_decompose=None, save_path=None):
    if (X.shape[1] != 1) or (y.shape[1] != 1):
        raise ValueError("Expected a 2D array for X and y.")

    if y_decompose is None:
        y_decompose = __ceemdan_decompose(y)
    
    
    solutions = []
    for decomp in y_decompose:
        y = np.c_[decomp]
        model.fit(X, y)
        sol = model.sympy()
        solutions.append(sol)

    if save_path:
        with open(save_path, "wb") as file:
            pickle.dump(solutions, file)
    