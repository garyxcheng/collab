from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset
import numpy as np
from scipy.linalg import sqrtm
import pandas as pd
import jax
import jax.numpy as jnp

### CRITICAL HELPERS ###
def make_projection_permutation_matrix(parent_df_columns_list, child_df_columns_list):
    parent_df_col_indices, other_col_indices = make_projection_permutation(parent_df_columns_list, child_df_columns_list)
    return np.eye(len(parent_df_columns_list))[parent_df_col_indices], np.eye(len(parent_df_columns_list))[other_col_indices]

def make_projection_permutation(parent_df_columns, child_df_columns):
    assert set(child_df_columns).issubset(set(parent_df_columns))
    if isinstance(parent_df_columns, pd.Index) or isinstance(parent_df_columns, np.ndarray):
        parent_df_columns = parent_df_columns.tolist()
    if isinstance(child_df_columns, pd.Index) or isinstance(child_df_columns, np.ndarray):
        child_df_columns = child_df_columns.tolist()
    parent_df_col_indices = [parent_df_columns.index(child_column) for child_column in child_df_columns]
    other_col_indices = [parent_df_columns.index(col) for col in parent_df_columns if col not in child_df_columns]
    return parent_df_col_indices, other_col_indices

def inverse_perm(perm):
    return np.argsort(perm)

def compute_covariance_matrix(dataset):
    if isinstance(dataset, FolktablesDataset):
        features = PandasToNumpyTransform()(dataset.features)
        return np.cov(features.T), np.mean(features, axis=0), dataset.columns
    elif isinstance(dataset, GaussianDataset):
        return dataset.cov, dataset.mean, dataset.columns
    else:
        raise Exception
    
    

### DATASET CLASS ###
class GaussianDataset:
    def __init__(self, theta, cov, sigma, columns, seed=0, transforms=None):
        self.theta = theta
        assert len(theta) == len(cov)
        self.cov = cov
        self.mean = np.zeros(len(cov))
        self.relevant_cov = cov[columns][:, columns]
        self.relevant_mean = self.mean[columns]
        # take square root of matrix
        self.relevant_cov_half = sqrtm(self.relevant_cov)
        self.cov_half = sqrtm(self.cov)
        self.sigma=sigma
        self.d = len(columns)
        self.all_d = len(cov)
        self.columns = columns
        self.seed = seed
        self.transforms = transforms

    def get_n_samples_numpy(self, n):
        np.random.seed(self.seed)
        X = np.random.normal(0, 1, (n, self.all_d)) @ self.cov_half
        # X = np.random.normal(0, 1, (n, self.d)) @ self.relevant_cov_half
        y = X @ self.theta + np.random.normal(0, self.sigma, n)
        return X[:, self.columns], y, self.columns

    def shuffle(self, seed):
        self.seed=seed

class FolktablesDataset(Dataset):
    def __init__(self, name, features, labels, columns, transforms=None):
        self.name = name
        self.features = features
        self.columns = columns
        self.labels = labels
        self.transforms = transforms
        assert len(self.features) == len(self.labels)
        assert np.all([self.features.columns[i] == self.columns[i] for i in range(len(self.columns))])
    
    def __len__(self):
        assert len(self.features) == len(self.labels)
        return len(self.features)
    
    def __getitem__(self, idx):
        data =self.features.iloc[idx]
        if self.transforms:
            data = self.transforms(data)
        return data, self.labels.iloc[idx], self.columns
    
    def shuffle(self, seed=0):
        self.features = self.features.sample(frac=1, random_state=seed)
        self.labels = self.labels.sample(frac=1, random_state=seed)
    
    def get_n_samples_numpy(self, n):
        return self.features[:n].values, self.labels[:n].values, self.columns
    
    def copy(self):
        return FolktablesDataset(self.name, self.features.copy(), self.labels.copy(), self.columns.copy(), transforms=self.transforms)
    
    def concat(self, other):
        assert np.all([self.features.columns[i] == other.features.columns[i] for i in range(len(self.columns))])
        assert np.all([self.columns[i] == other.columns[i] for i in range(len(self.columns))])
        features = pd.concat([self.features, other.features])
        labels = pd.concat([self.labels, other.labels])
        assert len(self.features) == len(self.labels)
        return FolktablesDataset(self.name, features, labels, self.columns, transforms=self.transforms)
    
    def __add__(self, other):
        return self.concat(other)
    
def hacky_cov_mse_fn(pred, target, cov=None):
    if cov is None:
        return mse_fn(pred, target)
    v = pred-target
    return v.T @ cov @ v / len(v)

def mse_fn(pred, target):
    return np.mean((pred-target)**2)
def abse_fn(pred, target):
    return np.mean(np.abs(pred-target))
def boolerr_fn(pred, target):
    return np.mean(np.abs(pred - target) > 0.5)

def compute_error(ds, model, metric, num_samples=None, imperfect=False):
    if isinstance(ds, FolktablesDataset):
        num_samples = len(ds) if num_samples is None else num_samples
        X, y, columns = ds.get_n_samples_numpy(num_samples)
        ypred = model.predict(X, X_columns=columns)
        return metric(y, ypred)
    elif isinstance(ds, GaussianDataset) and imperfect:
        num_samples = len(ds) if num_samples is None else num_samples
        X, y, columns = ds.get_n_samples_numpy(num_samples)
        ypred = model.predict(X, X_columns=columns)
        return metric(y, ypred)
    elif isinstance(ds, GaussianDataset) and metric == hacky_cov_mse_fn:
        htheta = np.zeros_like(ds.theta)
        htheta[model.columns] = model.weights
        return metric(htheta, ds.theta, cov=ds.cov) * len(ds.theta)
    elif isinstance(ds, GaussianDataset):
        htheta = np.zeros_like(ds.theta)
        htheta[model.columns] = model.weights
        return metric(htheta, ds.theta) * len(ds.theta)
    else:
        raise Exception

### TRANSFORMS ###
    
class PandasToNumpyTransform:
    def __call__(self, data):
        return np.array(data)

def numpy_collate_fn(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate_fn(samples) for samples in transposed]
    else:
        return np.array(batch)

### MODELS ###

class LinearRegressionModel:
    def __init__(self):
        self.weights = None
        self.residuals = None   
        self.rank = None
        self.singular_values = None
        self.train_X = None
        self.train_y = None
        self.columns = None

    def fit(self, X, y, columns):
        self.columns = columns
        self.train_X = X
        self.train_y = y
        self.weights, self.residuals, self.rank, self.singular_values = np.linalg.lstsq(X, y, rcond=None)
    
    def predict(self, X, X_columns=None, cov_mat=None):
        if X_columns is not None:
            perm_plus, perm_minus = make_projection_permutation(X_columns, self.columns)
            perm = perm_plus + perm_minus            
            X = np.atleast_2d(X)[:, perm_plus]
        
        if cov_mat is not None:
            weight_plus = self.weights[perm_plus]
            weight_minus = self.weights[perm_minus]
            Sig_iplus = cov_mat[perm_plus, :][:, perm_plus]
            Sig_ipm = cov_mat[perm_plus, :][:, perm_minus]
            # T = np.hstack((np.eye(len(model.columns)), np.linalg.inv(Sig_iplus) @ Sig_ipm))[:, inv_perm] @ self.weight
            new_weight = weight_plus + np.linalg.inv(Sig_iplus) @ Sig_ipm @ weight_minus
            return np.atleast_2d(X) @ new_weight
        else:
            return np.atleast_2d(X) @ self.weights

class ImputedLinearRegressionModel(LinearRegressionModel):
    def __init__(self, cov_all_X, columns):
        super().__init__()
        self.cov_all_X = cov_all_X
        self.columns = columns
        self.reweight_list = []

    def impute(self, X, y, X_columns, reweight=1.0):
        perm_plus, perm_minus = make_projection_permutation(self.columns, X_columns)
        perm = perm_plus + perm_minus
        inv_perm = inverse_perm(perm)
        Sig_iplus = self.cov_all_X[perm_plus][:, perm_plus]
        Sig_imp = self.cov_all_X[perm_minus][:, perm_plus]

        A = Sig_imp @ np.linalg.inv(Sig_iplus)
        X_minus = X @ A.T
        X = np.hstack([X, X_minus])[:, inv_perm]

        X = X * np.sqrt(reweight)
        y = y * np.sqrt(reweight)
        self.reweight_list.append(reweight)

        if self.train_X is None:
            self.train_X = X
        else:
            self.train_X = np.vstack([self.train_X, X])
        if self.train_y is None:
            self.train_y = y
        else:
            self.train_y = np.concatenate([self.train_y, y])
    
    def fit(self):
        self.weights, self.residuals, self.rank, self.singular_values = np.linalg.lstsq(self.train_X, self.train_y, rcond=None)

class LogisticRegressionModel:
    def __init__(self, fit_intercept=False, **kwargs):
        self.sklearn_model = LogisticRegression(fit_intercept=fit_intercept, **kwargs)
        self.train_X = None
        self.train_y = None
        self.columns = None

    @property
    def weights(self):
        return self.sklearn_model.coef_.flatten()
    
    def fit(self, X, y, columns):
        self.columns = columns
        self.train_X = X
        self.train_y = y
        self.sklearn_model.fit(X, y.flatten())
    
    def predict(self, X, X_columns=None):
        if X_columns is not None:
            perm_plus, _ = make_projection_permutation(X_columns, self.columns)
            X = np.atleast_2d(X)[:, perm_plus]
        return self.sklearn_model.predict(X)

    # def impute_then_fit(self, X, y, columns, cov_all_X, all_columns):
    #     perm_plus, perm_minus = make_projection_permutation(all_columns, columns)
    #     perm = perm_plus + perm_minus
    #     inv_perm = inverse_perm(perm)
    #     Sig_iplus = cov_all_X[perm_plus][:, perm_plus]
    #     Sig_imp = cov_all_X[perm_minus][:, perm_plus]

    #     A = Sig_imp @ np.linalg.inv(Sig_iplus)
    #     X_minus = X @ A.T
    #     X = np.hstack([X, X_minus])[:, inv_perm]

    #     self.columns = all_columns
    #     self.train_X = X
    #     self.train_y = y
    #     self.weights, self.residuals, self.rank, self.singular_values = np.linalg.lstsq(self.train_X, self.train_y, rcond=None)

class ImputedLogisticRegressionModel(LogisticRegressionModel):
    def __init__(self, cov_all_X, columns):
        super().__init__()
        self.cov_all_X = cov_all_X
        self.columns = columns

    def impute(self, X, y, X_columns):
        perm_plus, perm_minus = make_projection_permutation(self.columns, X_columns)
        perm = perm_plus + perm_minus
        inv_perm = inverse_perm(perm)
        Sig_iplus = self.cov_all_X[perm_plus][:, perm_plus]
        Sig_imp = self.cov_all_X[perm_minus][:, perm_plus]

        A = Sig_imp @ np.linalg.inv(Sig_iplus)
        X_minus = X @ A.T
        X = np.hstack([X, X_minus])[:, inv_perm]

        if self.train_X is None:
            self.train_X = X
        else:
            self.train_X = np.vstack([self.train_X, X])
        if self.train_y is None:
            self.train_y = y
        else:
            self.train_y = np.concatenate([self.train_y, y])
    
    def fit(self):
        self.sklearn_model.fit(self.train_X, self.train_y)

    # # TODO make this a seperate function.
    # def loss_fn(param):
    #     total = 0
    #     for idx in range(len(self.model_list)):
    #         perm_plus = self.perm_plus_list[idx]
    #         perm_minus = self.perm_minus_list[idx]
    #         v = param[perm_plus] + self.D_list[idx] @ param[perm_minus] - self.model_list[idx].weights
    #         total += v.T @ self.W_list[idx] @ v / self.losses_list[idx]
    #     return total
    # def generate_D_and_W_lists(self):
    #     T_list = []
    #     D_list = []
    #     W_list = []
    #     perm_plus_list = []
    #     perm_minus_list = []
    #     for model in self.model_list:
    #         perm_plus, perm_minus = make_projection_permutation(self.all_columns, model.columns)
    #         perm = perm_plus + perm_minus
    #         inv_perm = inverse_perm(perm)

    #         Sig_iplus = self.cov_matrix[perm_plus, :][:, perm_plus]
    #         Sig_ipm = self.cov_matrix[perm_plus, :][:, perm_minus]
    #         D_list.append(np.linalg.inv(Sig_iplus) @ Sig_ipm)
    #         T = np.hstack(np.eye(len(model.columns)), np.linalg.inv(Sig_iplus) @ Sig_ipm)[inv_perm]
    #         T_list.append(T)
    #         W_list.append(Sig_iplus)
    #         perm_plus_list.append(perm_plus)
    #         perm_minus_list.append(perm_minus)
    #     return D_list, W_list, perm_plus_list, perm_minus_list

    # def generate_Qs(self):
    #     Q_list = []
    #     for model in self.model_list:
    #         # proj_plus, proj_minus = make_projection_permutation_matrix(self.all_columns, model.columns)
    #         perm_plus, perm_minus = make_projection_permutation(self.all_columns, model.columns)
    #         perm = perm_plus + perm_minus
    #         # Sig_i = proj @ self.cov_matrix @ proj.T
    #         # Sig_iplus = proj_plus @ self.cov_matrix @ proj_plus.T
    #         # Sig_imp = proj_minus @ self.cov_matrix @ proj_plux.T
    #         # Sig_ipm = Sig_imp.T
    #         # Sig_i[proj_plus.shape[0]:, proj_plus.shape[0]:] = bottom_right_corner

    #         Sig_iplus = self.cov_matrix[perm_plus, :][:, perm_plus]
    #         Sig_imp = self.cov_matrix[perm_minus, :][:, perm_plus]
    #         bottom_right_corner = Sig_imp @ np.linalg.inv(Sig_iplus) @ Sig_imp.T

    #         Sig_i = self.cov_matrix[perm, :][:, perm]
    #         Sig_i[len(perm_plus):, len(perm_plus):] = bottom_right_corner

    #         Q_list.append(Sig_i)
    #     return Q_list

def NaiveAggregator_loss_fn(ensemble_weights, model_list, X, y, X_columns):
    pred_list = [model.predict(X, X_columns=X_columns) for model in model_list]
    pred_arr = jnp.stack(pred_list, axis=0)
    avg_pred = jnp.average(pred_arr, weights=ensemble_weights, axis=0)
    return jnp.mean((avg_pred - y) ** 2)

class NaiveAggregator:
    def __init__(self, all_columns, model_list):
        self.all_columns = all_columns
        self.columns = all_columns
        self.model_list = model_list
        self.ensemble_weights = np.ones(len(model_list)) / len(model_list)

    def predict(self, X, X_columns=None):
        if X_columns is None:
            X_columns = self.columns
        return np.average([model.predict(X, X_columns=X_columns) for model in self.model_list], weights=self.ensemble_weights, axis=0)

    def fit(self, X, y, X_columns, step_size=0.01, num_steps=100):
        if X_columns is None:
            X_columns = self.columns
        ensemble_weights = jnp.array(self.ensemble_weights)
        X, y = jnp.array(X), jnp.array(y)
        grad_fn = jax.grad(NaiveAggregator_loss_fn, argnums=0)
        for t in range(num_steps):
            ensemble_weights = ensemble_weights - step_size * grad_fn(ensemble_weights, self.model_list, X, y, X_columns)
        self.ensemble_weights = ensemble_weights

class MyModelAggregator:
    def fit(self, all_columns, model_list, cov_matrix, losses_list):
        assert np.all([model.columns is not None for model in model_list])
        self.all_columns = all_columns
        self.columns = all_columns
        self.model_list = model_list
        self.cov_matrix = cov_matrix
        self.losses_list = losses_list
        self.T_list, self.W_list, self.Q_list = self.generate_T_W_Q_lists()
        b = 0
        for i in range(len(self.model_list)):
            b += self.T_list[i].T @ self.W_list[i] @ self.model_list[i].weights / self.losses_list[i]
            # print(b)
        
        a = 0
        for i in range(len(self.model_list)):
            a += self.Q_list[i] / self.losses_list[i]
            # print(a)
        
        self.weights = np.linalg.solve(a, b)
    
    # def predict(self, X):
    #     return np.atleast_2d(X) @ self.weights
    def predict(self, X, X_columns=None, cov_mat=None):
        if X_columns is not None:
            perm_plus, perm_minus = make_projection_permutation(X_columns, self.columns)
            perm = perm_plus + perm_minus            
            X = np.atleast_2d(X)[:, perm_plus]
        
        if cov_mat is not None:
            weight_plus = self.weights[perm_plus]
            weight_minus = self.weights[perm_minus]
            Sig_iplus = cov_mat[perm_plus, :][:, perm_plus]
            Sig_ipm = cov_mat[perm_plus, :][:, perm_minus]
            # T = np.hstack((np.eye(len(model.columns)), np.linalg.inv(Sig_iplus) @ Sig_ipm))[:, inv_perm] @ self.weight
            new_weight = weight_plus + np.linalg.inv(Sig_iplus) @ Sig_ipm @ weight_minus
            return np.atleast_2d(X) @ new_weight
        else:
            return np.atleast_2d(X) @ self.weights
    
    def generate_T_W_Q_lists(self):
        T_list = []
        W_list = []
        Q_list = []
        for model in self.model_list:
            perm_plus, perm_minus = make_projection_permutation(self.all_columns, model.columns)
            perm = perm_plus + perm_minus
            inv_perm = inverse_perm(perm)

            Sig_iplus = self.cov_matrix[perm_plus, :][:, perm_plus]
            Sig_ipm = self.cov_matrix[perm_plus, :][:, perm_minus]
            T = np.hstack((np.eye(len(model.columns)), np.linalg.inv(Sig_iplus) @ Sig_ipm))[:, inv_perm]

            T_list.append(T)
            W_list.append(Sig_iplus)

            Sig_imp = Sig_ipm.T
            bottom_right_corner = Sig_imp @ np.linalg.inv(Sig_iplus) @ Sig_imp.T
            Sig_i = self.cov_matrix[perm, :][:, perm]
            Sig_i[len(perm_plus):, len(perm_plus):] = bottom_right_corner

            Q_list.append(Sig_i[inv_perm, :][:, inv_perm])
        return T_list, W_list, Q_list
    
class MyLogisticModelAggregator(MyModelAggregator):
    def predict(self, X):
        # model = LogisticRegression(fit_intercept=False)
        # model.coef_ = self.weights
        # model.intercept_ = 0.0
        # return model.predict(X)
        yhat = np.atleast_2d(X) @ self.weights
        return np.where(yhat > 0, 1, 0)



### ARCHIVE ###

# class SingleShuffleDataLoader(DataLoader):
#     def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
#         super().__init__(dataset, batch_size=batch_size, shuffle=False, **kwargs)
#         self.shuffle = shuffle
#         self.first_iter = True
#         self.indices = None

#     def __iter__(self):
#         if self.shuffle and self.first_iter:
#             self.first_iter = False
#             self.indices = list(RandomSampler(self.dataset))
#             return self._get_batches(self.indices)
#         else:
#             return self._get_batches(self.indices)

#     def _get_batches(self, indices):
#         batch = []
#         for idx in indices:
#             batch.append(idx)
#             if len(batch) == self.batch_size:
#                 yield [self.dataset[i] for i in batch]
#                 batch = []