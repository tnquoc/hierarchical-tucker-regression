import numpy as np
import tensorly as tl
from tensorly.base import partial_tensor_to_vec, vec_to_tensor, partial_unfold
from tensorly.tenalg import kronecker

from base_hierarchical_tucker import HierarchicalTuckerTensor


class HierarchicalTuckerRegressor:
    """Hierarchical Tucker tensor regression

        Learns a low rank Hierarchical Tucker weight for the regression

    Parameters
    ----------
    dims : int list
        list values of each dimesion of tensor
    ht_ranks : int list
        rank of each node in binary tree for Hierarchical Tucker format
    tol : float
        convergence value
    n_iter_max : int, optional, default is 100
        maximum number of iteration
    random_state : None, int or RandomState, optional, default is None
    verbose : int, default is 1
        level of verbosity
    """

    def __init__(self, dims, ht_ranks, tol=10e-7, n_iter_max=100, random_state=None, verbose=1):
        self.dims = dims
        self.ht_ranks = ht_ranks
        self.tol = tol
        self.n_iter_max = n_iter_max
        self.random_state = random_state
        self.verbose = verbose

        # create a Hierarchical Tucker format of tensor
        self.HTuckerTensor = HierarchicalTuckerTensor(dims, ht_ranks)

    def get_params(self, **kwargs):
        """Returns a dictionary of parameters
        """
        params = ['dims', 'ht_ranks', 'tol',
                  'n_iter_max', 'random_state', 'verbose']
        return {param_name: getattr(self, param_name) for param_name in params}

    def set_params(self, **parameters):
        """Sets the value of the provided parameters"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        """
        Fits the model to the data (X, y)

        Parameters
        ----------
        X : ndarray of shape (n_samples, N1, ..., NS)
            tensor data
        y : array of shape (n_samples)
            labels associated with each sample

        Returns
        -------
        self
        """
        # Norm of the weight tensor at each iteration
        norm_W = []

        # properties of Hierarchical Tucker format
        children = self.HTuckerTensor.children
        dim2ind = self.HTuckerTensor.dim2ind
        tree_structure = self.HTuckerTensor.tree_structure
        is_leaf = self.HTuckerTensor.is_leaf()

        U = self.HTuckerTensor.U
        B = self.HTuckerTensor.B
        U_cal = [u for u in U if len(u)]

        idx_level = self.HTuckerTensor.get_level()
        distinct_level = list(dict.fromkeys(idx_level))
        reverse_distinct_level = distinct_level[::-1]
        root_level = 0
        highest_level = distinct_level[-1]
        factor_in_tree_structure_by_level = self.HTuckerTensor.factor_in_tree_structure_by_level

        for iteration in range(self.n_iter_max):

            # Optimize leaf nodes
            L = B[0]

            for i in range(len(U_cal)):
                cur_level = idx_level[dim2ind[i]]
                leaf_ht_ranks = [self.ht_ranks[j] for j in dim2ind]
                L = L.reshape(int(np.prod(leaf_ht_ranks) /
                                  leaf_ht_ranks[i]), leaf_ht_ranks[i])

                phi = partial_tensor_to_vec(tl.dot(partial_unfold(X, tree_structure[dim2ind[i]][0] - 1),
                                                   tl.dot(kronecker(U_cal, skip_matrix=i), L)))
                A = tl.dot(phi.T, phi)
                b = tl.dot(phi.T, y)
                U_i = vec_to_tensor(tl.dot(np.linalg.pinv(
                    A), b), (U_cal[i].shape[0], U_cal[i].shape[1]))
                U_cal[i] = U_i
                U[dim2ind[i]] = U_i
                factor_in_tree_structure_by_level[cur_level][dim2ind[i]] = U_i

            # Optimise interior nodes
            for level in reverse_distinct_level:
                if level == highest_level:
                    continue

                if level == root_level and len(distinct_level) == 2:
                    H = tl.dot(kronecker(U_cal).T, partial_tensor_to_vec(X).T)
                    phi = H.T
                    A = tl.dot(phi.T, phi)
                    b = tl.dot(phi.T, y)
                    B_i = vec_to_tensor(tl.dot(np.linalg.pinv(
                        A), b), (B[0].shape[0], B[0].shape[1]))
                    B[0] = B_i
                    factor_in_tree_structure_by_level[level][0] = B_i
                    break

            weight_tensor_ = kronecker(U_cal)
            for level in reverse_distinct_level:
                if level == highest_level:
                    continue
                if len(factor_in_tree_structure_by_level[level]["extra"]):
                    val = factor_in_tree_structure_by_level[level]["extra"] + list(
                        factor_in_tree_structure_by_level[level].values())[1:-1]
                else:
                    val = list(factor_in_tree_structure_by_level[level].values())[1:]
                weight_tensor_ = tl.dot(weight_tensor_, kronecker(val))
            weight_tensor_ = vec_to_tensor(weight_tensor_, tuple(X.shape[1:]))

            norm_W.append(tl.norm(weight_tensor_, 2))

            # Convergence check
            if iteration > 1:
                weight_evolution = abs(norm_W[-1] - norm_W[-2]) / norm_W[-1]

                if weight_evolution <= self.tol:
                    if self.verbose:
                        print('\nConverged in {} iterations'.format(iteration))
                    break

        self.weight_tensor_ = weight_tensor_
        self.n_iterations_ = iteration + 1
        self.norm_W_ = norm_W

        return self
