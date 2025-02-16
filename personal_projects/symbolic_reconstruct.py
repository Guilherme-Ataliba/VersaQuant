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
        

    def reconstruct(self, local_path=None):
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



class ReconstructAnalyse():
    def __init__(self):
        self.solutions_df = None

    def createSolutions(self, dir, file_name):
        files = os.listdir(dir)
        size = len(file_name)

        ranges = []
        indexes = []
        solutions_df = []

        for file in files:
            if (file[0:size] == file_name):
                range_ = re.findall(r"-[\d\.]*-", file)[0][1:-1]
                index = re.findall(r"-\d*\.pkl", file)[0][1:-4]

                ranges.append(range_)
                indexes.append(index)

                with open(os.path.join(dir, file), "rb") as file_:
                    solution_ = pickle.load(file_)

                    solu_df = pd.DataFrame(solution_)
                
                solu_df["Range"] = range_
                solu_df["Iter"] = index

                solutions_df.append(solu_df)

        final_df = pd.concat(solutions_df)
        self.solutions_df = final_df
        return final_df
    
    @staticmethod
    def time_train_test_split(X, y, train_size=0.8):
        X_ = X.reshape(-1, )
        y_ = y.reshape(-1, )

        size = X_.size
        div_id = int(size*train_size)

        X_train, X_test = X_[:div_id], X_[div_id:]
        y_train, y_test = y_[:div_id], y_[div_id:]

        return X_train, y_train, X_test, y_test
            
    def createSum(self, X, y, solutions_df=None, sympy_var=[smp.symbols("x0")], dir=None, file_name=None):
        if solutions_df is None:
            if (self.solutions_df is None):
                if (dir is not None) and (file_name is not None):
                    self.createSolutions(dir, file_name)
                else:
                    raise ValueError("You must run createSolutions first or inform a solution dataframe")
            solutions_df = self.solutions_df.copy()

        solutions_all = []
        for range_ in solutions_df["Range"].unique():
            X_train, y_train, X_test, y_test = self.time_train_test_split(X, y, float(range_))

            sols_ = solutions_df[solutions_df["Range"] == range_]

            solutions = []
            for iter_ in sols_["Iter"].unique():
                sols__ = sols_[sols_["Iter"] == iter_]
                sum_expr = smp.lambdify(sympy_var, np.sum(sols__["Expr"].values))
                
                y_pred = sum_expr(X_test)

                solutions_dict = {
                    "Iter": iter_,
                    "Range": range_,
                    "SumExpr": sum_expr,
                    "y_pred": y_pred,
                    "y_test": y_test
                }
                solutions.append(solutions_dict)

            solutions_all.append(pd.DataFrame(solutions))

        solutions_df = pd.concat(solutions_all).reset_index(drop=True)
        self.summ_df = solutions_df
        return solutions_df
    
    def cosine_similarity(self, summ_df=None, n_points=None) -> float:
        if summ_df is None:
            if self.summ_df is None:
                raise ValueError("Yous run createSum or give a summ dataframe")        
            summ_df = self.summ_df

        cosine_values = []
        for _, row in summ_df.iterrows():

            if n_points is None:
                actual = np.array(row["y_test"])
                predicted = np.array(row["y_pred"])
            else:
                actual = np.array(row["y_test"][0:n_points])
                predicted = np.array(row["y_pred"][0:n_points])
            
            dot_product = np.dot(actual, predicted)
            norm_actual = np.linalg.norm(actual)
            norm_predicted = np.linalg.norm(predicted)
            
            # Avoid division by zero
            if norm_actual == 0 or norm_predicted == 0:
                return 0.0  # If either vector is zero, similarity is undefined (return 0)
            
            cosine_value = {
                "Iter": row.Iter,
                "Range": row.Range,
                "CosineSimilarity": dot_product / (norm_actual * norm_predicted)
            }
            
            cosine_values.append(cosine_value)
        
        return pd.DataFrame(cosine_values)


    def angle_similarity(self, summ_df = None, n_points=None, derivative=True) -> float:
        if summ_df is None:
            if self.summ_df is None:
                raise ValueError("Yous run createSum or give a summ dataframe")        
            summ_df = self.summ_df
        
        angle_values = []
        for _, row in summ_df.iterrows():
            if n_points is None:
                actual = np.array(row["y_test"])
                predicted = np.array(row["y_pred"])
            else:
                actual = np.array(row["y_test"][0:n_points])
                predicted = np.array(row["y_pred"][0:n_points])
            
            # Cetenring the start of both vectors at the origin
            actual_zero_centered = actual - actual[0]
            predicted_zero_centered = predicted - predicted[0]
            
            if derivative:
                angle_actual = np.arctan2(np.gradient(actual_zero_centered), 1)
                angle_predicted = np.arctan2(np.gradient(predicted_zero_centered), 1)
            else:
                angle_actual = np.arctan2(actual_zero_centered, 1)
                angle_predicted = np.arctan2(predicted_zero_centered, 1)

            direction = (np.mean(angle_actual) *np.mean(angle_predicted))/np.abs(np.mean(angle_actual) *np.mean(angle_predicted))

            angle_value = {
                "Iter": row.Iter,
                "Range": row.Range,
                "AngleReal": angle_actual.mean(),
                "AnglePred": angle_predicted.mean(),
                "Agreement": direction,
                "PredDirection": angle_predicted.mean()/(np.abs(angle_predicted.mean()))
            }

            angle_values.append(angle_value)

            
            
        return pd.DataFrame(angle_values)
        
    def similarity_analisis(self, summ_df=None, n_points=None, derivative=True):
        if summ_df is None:
            if self.summ_df is None:
                raise ValueError("Yous run createSum or give a summ dataframe")        
            summ_df = self.summ_df

        angle_similarity = self.angle_similarity(summ_df, n_points, derivative).drop(["Iter", "Range"], axis=1)
        cosine_similarity = self.cosine_similarity(summ_df, n_points)

        return pd.concat([cosine_similarity, angle_similarity], axis=1)
    

    def conclusion_analysis(self, summ_df=None, n_points=None, derivative=True):
        if summ_df is None:
            if self.summ_df is None:
                raise ValueError("Yous run createSum or give a summ dataframe")        
            summ_df = self.summ_df

        similarity_df = self.similarity_analisis(summ_df, n_points, derivative)

        infos = {}
        for range_ in similarity_df.Range.unique():
            df_ = similarity_df[similarity_df.Range == range_]
            mode = df_.PredDirection.mode()[0]

            prob = df_.PredDirection.value_counts().loc[mode]/(df_.PredDirection.value_counts().loc[1.0] + df_.PredDirection.value_counts().loc[-1.0])

            cosine_mean = df_.CosineSimilarity.mean()

            info = {
                "Direction": mode,
                "Probility": prob,
                "CosineSimilarityMean": cosine_mean
            }

            infos[range_] = info
        
        df = pd.DataFrame(infos).T
        df.index.name = "Range"
        return df
