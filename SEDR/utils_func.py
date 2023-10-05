#
import os
import torch
import numpy as np
import scanpy as sc

def adata_preprocess(adata_vis, min_cells=50, min_counts=10, pca_n_comps=200):
    adata_vis.layers['count'] = adata_vis.X.toarray()
    sc.pp.filter_genes(adata_vis, min_cells=min_cells)
    sc.pp.filter_genes(adata_vis, min_counts=min_counts)

#     adata_vis.obs['mean_exp'] = adata_vis.X.toarray().mean(axis=1)
#     adata_vis.var['mean_exp'] = adata_vis.X.toarray().mean(axis=0)
#
#     # Load scRNA-seq data
#     adata_ref = sc.read_h5ad('/home/xuhang/disco_500t/Projects/spTrans/data/reference_data/GSE144136_DLPFC/raw/processed_raw.h5ad')
#     adata_ref.obs['mean_exp'] = adata_ref.X.toarray().mean(axis=1)
#     adata_ref.var['mean_exp'] = adata_ref.X.toarray().mean(axis=0)
#     common_genes = np.intersect1d(adata_vis.var.index, adata_ref.var.index)
#     adata_vis = adata_vis[:, common_genes]
#     adata_ref = adata_ref[:, common_genes]
#     adata_vis.var['ref_mean_exp'] = adata_ref.var['mean_exp']
#     adata_vis.var['ratio'] = np.log10(adata_vis.var['mean_exp'] / adata_vis.var['ref_mean_exp']+1)
#     adata_vis.var['selected'] = adata_vis.var['ratio'] < 1.5
#     remain_genes = adata_vis.var[adata_vis.var['selected']==True].index.tolist()
#     adata_vis = adata_vis[:, remain_genes]
#
#
    sc.pp.normalize_total(adata_vis, target_sum=1e6)
    # sc.pp.log1p(adata_vis)
    sc.pp.highly_variable_genes(adata_vis, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata_vis = adata_vis[:, adata_vis.var['highly_variable'] == True]
    sc.pp.scale(adata_vis)

    from sklearn.decomposition import PCA
    adata_X = PCA(n_components=pca_n_comps, random_state=42).fit_transform(adata_vis.X)
    adata_vis.obsm['X_pca'] = adata_X
    return adata_vis


def fix_seed(seed):
    import random
    import torch
    from torch.backends import cudnn

    #seed = 666
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
