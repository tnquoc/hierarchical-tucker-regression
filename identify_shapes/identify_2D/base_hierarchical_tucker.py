import numpy as np
import math


class BaseHierarchicalTuckerTensor:
    def __init__(self, dims: list, tree_type=None, children=None, dim2ind=None, U=None, B=None):
        if not dims:
            raise ValueError('Missing required argument dims.')
        if not isinstance(dims, list):
            raise ValueError('dims must be a vector of positive intergers.')
        if len(dims) == 0:
            raise ValueError('dims must not be empty.')
        self.children = np.zeros((2 * len(dims) - 1, 2))
        self.dim2ind = [0] * len(dims)
        self.tree_structure = []
        self.U = []
        self.B = []
        self.children, self.dim2ind, self.tree_structure = define_tree(
            dims, tree_type)
        if children and dim2ind and U and B:
            self.children = children
            self.dim2ind = dim2ind
            self.U = U
            self.B = B
        self.U = [None] * self.children.shape[0]
        self.B = [None] * self.children.shape[0]
        for i in range(self.children.shape[0]):
            if self.children[i, 0] == 0:
                self.U[i] = np.zeros((dims[self.tree_structure[i][0] - 1], 1))
                self.B[i] = []
            else:
                self.U[i] = []
                self.B[i] = 1

    def get_n_nodes(self):
        return self.children.shape[0]

    def is_leaf(self):
        leaves = [1 if val == 0 else 0 for val in self.children[:, 0]]
        return leaves

    def get_parents(self):
        parents = [0] * self.get_n_nodes()
        is_leaf = self.is_leaf()
        interior_nodes_idx = [i for i, x in enumerate(is_leaf) if x == 0]
        for i in interior_nodes_idx:
            parents[int(self.children[i, 0])] = i
            parents[int(self.children[i, 1])] = i
        return parents

    def get_level(self):
        level = [0] * self.get_n_nodes()
        ind = subtree(self.children, 0)
        parents = self.get_parents()
        for i in ind[1:]:
            level[int(i)] = level[parents[int(i)]] + 1
        return level


def define_tree(dims, tree_type='balance'):
    d = len(dims)
    children = np.zeros((2 * d - 1, 2))
    tree_structure = [None] * (2 * d - 1)
    tree_structure[0] = list(range(1, len(dims) + 1))
    n_nodes = 0
    i = 0
    while i <= n_nodes:
        if len(tree_structure[i]) == 1:
            children[i, :] = np.array([0, 0])
        else:
            i_left = n_nodes + 1
            i_right = n_nodes + 2
            n_nodes = n_nodes + 2
            children[i, :] = np.array([i_left, i_right])
            if tree_type == 'first_separeate' and i == 0:
                tree_structure[i_left] = tree_structure[i][:1]
                tree_structure[i_right] = tree_structure[i][1:]
            elif tree_type == 'first_pair_separeate' and i == 0:
                tree_structure[i_left] = tree_structure[i][:2]
                tree_structure[i_right] = tree_structure[i][2:]
            elif tree_type == 'TT':
                tree_structure[i_left] = tree_structure[i][:1]
                tree_structure[i_right] = tree_structure[i][1:]
            else:
                tree_structure[i_left] = tree_structure[i][:math.ceil(len(tree_structure[i]) / 2)]
                tree_structure[i_right] = tree_structure[i][math.ceil(len(tree_structure[i]) / 2):]
        i += 1
    dim2ind = np.where(children[:, 0] == 0)[0]
    return children, dim2ind, tree_structure


def subtree(children, idx):
    if isinstance(children, BaseHierarchicalTuckerTensor):
        children = children.children
    sub_idx = [idx]
    i = 0
    while i < len(sub_idx) < children.shape[0]:
        if not np.all(children[int(sub_idx[i]), :] == [0, 0]):
            if i == 0:
                sub_idx = [sub_idx[0]] + list(children[sub_idx[0], :])
            else:
                sub_idx = sub_idx[:i + 1] + list(children[int(sub_idx[i]), :]) + sub_idx[i:]
            sub_idx = list(dict.fromkeys(sub_idx))
        i += 1
    return sub_idx


class HierarchicalTuckerTensor(BaseHierarchicalTuckerTensor):
    def __init__(self, dims, ht_ranks, tree_type=None, children=None, dim2ind=None, U=None, B=None):
        super().__init__(dims, tree_type, children, dim2ind, U, B)
        # BaseHierarchicalTuckerTensor(
        #     self, dims, tree_type, children, dim2ind, U, B)
        self._validate(dims, ht_ranks)
        self.dims = dims
        self.ht_ranks = ht_ranks
        is_leaf = self.is_leaf()
        for i in range(self.get_n_nodes()):
            if is_leaf[i]:
                self.U[i] = np.random.rand(self.U[i].shape[0], ht_ranks[i])
            else:
                i_left = int(self.children[i, 0])
                i_right = int(self.children[i, 1])
                self.B[i] = np.random.rand(
                    ht_ranks[i_left] * ht_ranks[i_right], ht_ranks[i])

        self.factor_in_tree_structure_by_level = self._create_tree_structure_by_level()

    @staticmethod
    def _validate(dims, ht_ranks):
        if not ht_ranks:
            raise ValueError('Missing required argument ht_ranks.')
        if ht_ranks and len(ht_ranks) != 2 * len(dims) - 1:
            raise ValueError(
                'ht_ranks must be a vector of 2 * len(dims) - 1 positive integers.')
        if ht_ranks[0] != 1:
            raise ValueError(
                'The first element of ht_ranks must be 1 because it is the rank of the root.')

    def _create_tree_structure_by_level(self):
        is_leaf = self.is_leaf()
        idx_level = self.get_level()
        distinct_level = list(dict.fromkeys(idx_level))
        highest_level = distinct_level[-1]
        factor_in_tree_structure_by_level = {}
        for level in distinct_level:
            factor_in_tree_structure_by_level[level] = {}
            factor_in_tree_structure_by_level[level]["extra"] = []

            if level == highest_level:
                indices = [i for i, x in enumerate(
                    is_leaf) if x == 1 and idx_level[i] == highest_level]
                for i in indices:
                    factor_in_tree_structure_by_level[level][i] = self.U[i]
                continue

            indices = [i for i, x in enumerate(idx_level) if x == level]
            if level == highest_level - 1:
                for i in indices:
                    if is_leaf[i]:
                        factor_in_tree_structure_by_level[level][i] = self.U[i]
                        factor_in_tree_structure_by_level[level]["extra"].append(np.eye(
                            self.ht_ranks[i]))
                    else:
                        factor_in_tree_structure_by_level[level][i] = self.B[i]
            else:
                for i in indices:
                    factor_in_tree_structure_by_level[level][i] = self.B[i]

        return factor_in_tree_structure_by_level


# k = HierarchicalTuckerTensor([1, 2, 3, 4, 5], [1, 2, 2, 2, 2, 2, 2, 2, 2])
# print(k.children)
# print(k.dim2ind)
# print(k.tree_structure)
# print(k.U)
# print(k.B)
# print(k.is_leaf())
# print(k.factor_in_tree_structure_by_level)
# print(k.get_level(), k.get_n_nodes())
