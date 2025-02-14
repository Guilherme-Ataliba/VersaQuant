import pandas as pd
import numpy as np
import sympy as smp
from PyEMD import CEEMDAN
import pickle
import os
import re

class SymbolicReconstruct():

    def __init__(self, model, save_path=None):
        """
            Need to add general symbolic regression model
        """

        self.model = model
        self.multi_solutions = None

        if save_path:
            test_path = re.findall(r".*\/", save_path)[0] 
            if os.path.isdir(test_path):
                self.save_path = save_path
            else:
                raise NotADirectoryError(f"{save_path} is not a valid directory")
        
        

    def ceemdan_decompose(self, series, seed=None):
        if seed is not None:
            np.random.seed(seed)

        series = series.reshape(-1, )
        ceemdan = CEEMDAN()  # Criando o objeto CEEMDAN
        ceemdan.MAX_IMF = 5  # Set the maximum number of IMFs
        imfs = ceemdan(series)  # Decompondo a série
        return imfs
    
    def fit(self, X, y, y_decompose=None, seed=None):
        if (X.shape[1] != 1) or (y.shape[1] != 1):
            raise ValueError("Expected a 2D array for X and y.")
        
        self.X = X
        self.y = y

        if seed is not None:
            np.random.seed(seed)

        if y_decompose is None:
            self.decomposition = self.ceemdan_decompose(self.y, seed)
            self.y_decompose = self.decomposition.copy()

    def set_use_indexes(self, use_indexes):
        self.use_indexes = use_indexes
        if use_indexes is not None:
            self.y_decompose = self.y_decompose[use_indexes]
        

    def reconstruct(self, y_decompose=None, local_path=None):
        solutions = []
        for c, decomp in enumerate(self.y_decompose):
            y = np.c_[decomp]
            self.model.fit(self.X, y)
            expr = self.model.sympy()
            
            y_pred = self.model.predict(self.X)
            MSE = np.mean((y-y_pred)**2)

            solution = {
                "Expr": expr,
                "MSE": MSE,
                "EMF": self.use_indexes[c] if self.use_indexes is not None else c
            }

            solutions.append(solution)

        if local_path:
             with open(local_path, "wb") as file:
                pickle.dump(solutions, file)
        elif self.save_ath:
            with open(f"{self.save_path}", "wb") as file:
                pickle.dump(solutions, file)

        return solutions
    
    def multiple_reconstruct(self, n_times, add_path_name=None):
        
        solutions = []
        for n in range(n_times):
            print(f"Started Decomp - {n}")
            solution = self.reconstruct(local_path=f"{self.save_path}-{n}.pkl")
            solutions.append(solution)

        self.multi_solutions = solutions
        

        if self.save_path:
            if add_path_name:
                sol_path = re.findall(r".*\/", self.save_path)[0] + f"multi_reconstruct-{add_path_name}.pkl"
            else:
                sol_path = re.findall(r".*\/", self.save_path)[0] + f"multi_reconstruct.pkl"

            with open(sol_path, "wb") as file:
                pickle.dump(solutions, file)

    def fit_solutions(self, solutions=None, solutions_file=None):
        if solutions is not None:
            self.multi_solutions = solutions
        elif solutions_file is not None:
            with open(solutions_file, "rb") as file:
                self.multi_solutions = pickle.load(file)


    def get_best_reconstruction(self, solutions_file=None):
        """
            The idea is to look at every solution at pick the one with the lowest MSE overall
        """

        if solutions_file:
            with open(solutions_file, "rb") as file:
                self.multi_solutions = pickle.load(file)
        
        if self.multi_solutions is None:
            raise ValueError("You must run multiple solutions method or indicate a file that contains the solutions.")
    
        dfs = []
        for c in range(len(self.multi_solutions)):
            df_ = pd.DataFrame(self.multi_solutions[c])
            df_["Iter"] = c
            dfs.append(df_)

        multi_solutions_df = pd.concat(dfs)

        best_solutions = {}
        for emf in multi_solutions_df["EMF"].unique():
            multi_ = multi_solutions_df[multi_solutions_df["EMF"] == emf].reset_index(drop=True)
            best_row = multi_.loc[multi_["MSE"].idxmin()]

            best_sol = {
                "Expr": best_row.Expr,
                "MSE": best_row.MSE,
                "Iter": best_row.Iter
            }

            best_solutions[emf] = best_sol

        self.best_solutions = best_solutions
        return best_solutions



        