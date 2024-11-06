#
# import networkx as nx
import numpy as np
import torch
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph



##### generate n
def generate_adj_mat(adata, include_self=False, n=6):
    from sklearn import metrics
    assert 'spatial' in adata.obsm, 'AnnData object should provided spatial information'

    dist = metrics.pairwise_distances(adata.obsm['spatial'])

    # sample_name = list(adata.uns['spatial'].keys())[0]
    # scalefactors = adata.uns['spatial'][sample_name]['scalefactors']
    # adj_mat = dist <= scalefactors['fiducial_diameter_fullres'] * (n+0.2)
    # adj_mat = adj_mat.astype(int)

    # n_neighbors = np.argpartition(dist, n+1, axis=1)[:, :(n+1)]
    # adj_mat = np.zeros((len(adata), len(adata)))
    # for i in range(len(adata)):
    #     adj_mat[i, n_neighbors[i, :]] = 1

    adj_mat = np.zeros((len(adata), len(adata)))
    for i in range(len(adata)):
        n_neighbors = np.argsort(dist[i, :])[:n+1]
        adj_mat[i, n_neighbors] = 1

    if not include_self:
        x, y = np.diag_indices_from(adj_mat)
        adj_mat[x, y] = 0

    adj_mat = adj_mat + adj_mat.T
    adj_mat = adj_mat > 0
    adj_mat = adj_mat.astype(np.int64)

    return adj_mat


def generate_adj_mat_1(adata, max_dist):
    from sklearn import metrics
    assert 'spatial' in adata.obsm, 'AnnData object should provided spatial information'

    dist = metrics.pairwise_distances(adata.obsm['spatial'], metric='euclidean')
    adj_mat = dist < max_dist
    adj_mat = adj_mat.astype(np.int64)
    return adj_mat

##### normalze graph
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def mask_generator(adj_label, N=1):
    idx = adj_label.indices()
    cell_num = adj_label.size()[0]

    list_non_neighbor = []
    for i in range(0, cell_num):
        neighbor = idx[1, torch.where(idx[0, :] == i)[0]]
        n_selected = len(neighbor) * N

        # non neighbors
        total_idx = torch.range(0, cell_num-1, dtype=torch.float32)
        non_neighbor = total_idx[~torch.isin(total_idx, neighbor)]
        indices = torch.randperm(len(non_neighbor), dtype=torch.float32)
        random_non_neighbor = indices[:n_selected]
        list_non_neighbor.append(random_non_neighbor)

    x = adj_label.indices()[0]
    y = torch.concat(list_non_neighbor)

    indices = torch.stack([x, y])
    indices = torch.concat([adj_label.indices(), indices], axis=1)

    value = torch.concat([adj_label.values(), torch.zeros(len(x), dtype=torch.float32)])
    adj_mask = torch.sparse_coo_tensor(indices, value)

    return adj_mask


def graph_computing(pos, n):
    from scipy.spatial import distance
    list_x = []
    list_y = []
    list_value = []

    for node_idx in range(len(pos)):
        tmp = pos[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, pos, 'euclidean')
        res = distMat.argsort()
        # tmpdist = distMat[0, res[0][1:params.k + 1]]
        for j in np.arange(1, n + 1):
            list_x += [node_idx, res[0][j]]
            list_y += [res[0][j], node_idx]
            list_value += [1, 1]

    adj = sp.csr_matrix((list_value, (list_x, list_y)))
    adj = adj >= 1
    adj = adj.astype(np.float32)
    return adj


def graph_construction(adata, n=6, dmax=50, mode='KNN'):
    if mode == 'KNN':
        adj_m1 = generate_adj_mat(adata, include_self=False, n=n)
        # adj_m1 = graph_computing(adata.obsm['spatial'], n=n)
    else:
        adj_m1 = generate_adj_mat_1(adata, dmax)
    adj_m1 = sp.coo_matrix(adj_m1)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()

    # Some preprocessing
    adj_norm_m1 = preprocess_graph(adj_m1)
    adj_m1 = adj_m1 + sp.eye(adj_m1.shape[0])
    # adj_label_m1 = torch.FloatTensor(adj_label_m1.toarray())

    adj_m1 = adj_m1.tocoo()
    shape = adj_m1.shape
    values = adj_m1.data
    indices = np.stack([adj_m1.row, adj_m1.col])
    adj_label_m1 = torch.sparse_coo_tensor(indices, values, shape)

    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)

    # # generate random mask
    # adj_mask = mask_generator(adj_label_m1.to_sparse(), N)

    graph_dict = {
        "adj_norm": adj_norm_m1,
        "adj_label": adj_label_m1.coalesce(),
        "norm_value": norm_m1,
        # "mask": adj_mask
    }

    return graph_dict

def block_diag_sparse(*arrs):
        bad_args = [k for k in range(len(arrs)) if not (isinstance(arrs[k], torch.Tensor) and arrs[k].ndim == 2)]
        if bad_args:
            raise ValueError("arguments in the following positions must be 2-dimension tensor: %s" % bad_args)

        list_shapes = [a.shape for a in arrs]
        list_indices = [a.coalesce().indices().clone() for a in arrs]
        list_values = [a.coalesce().values().clone() for a in arrs]

        r_start = 0
        c_start = 0
        for i in range(len(arrs)):
            list_indices[i][0, :] += r_start
            list_indices[i][1, :] += c_start

            r_start += list_shapes[i][0]
            c_start += list_shapes[i][1]

        indices = torch.concat(list_indices, axis=1)
        values = torch.concat(list_values)
        shapes = torch.tensor(list_shapes).sum(axis=0)

        out = torch.sparse_coo_tensor(indices, values, (shapes[0], shapes[1]))

        return out


def combine_graph_dict(dict_1, dict_2):
    # TODO add adj_org
    tmp_adj_norm = block_diag_sparse(dict_1['adj_norm'], dict_2['adj_norm'])
    tmp_adj_label = block_diag_sparse(dict_1['adj_label'], dict_2['adj_label'])
    graph_dict = {
        "adj_norm": tmp_adj_norm.coalesce(),
        "adj_label": tmp_adj_label.coalesce(),
        "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']])
    }
    return graph_dict



# def combine_graph_dict(dict_1, dict_2):
#     # TODO add adj_org
#     tmp_adj_norm = torch.block_diag(dict_1['adj_norm'].to_dense(), dict_2['adj_norm'].to_dense())
#     tmp_adj_label = torch.block_diag(dict_1['adj_label'].to_dense(), dict_2['adj_label'].to_dense())
#     graph_dict = {
#         "adj_norm": tmp_adj_norm.to_sparse(),
#         "adj_label": tmp_adj_label.to_sparse(),
#         "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']])
#     }
#     return graph_dict