# Unsupervised Batch correction
import os
import torch
import argparse
import warnings
import numpy as np
import pandas as pd
import anndata
from src.graph_func import graph_construction, combine_graph_dict
from src.utils_func import mk_dir, adata_preprocess, load_ST_file
from src.SEDR_train import SEDR_Train
import matplotlib.pyplot as plt
import scanpy as sc

warnings.filterwarnings('ignore')
torch.cuda.cudnn_enabled = False
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device: ' + device)

# ################ Parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=10, help='parameter k in spatial graph')
parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                    help='graph distance type: euclidean/cosine/correlation')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--cell_feat_dim', type=int, default=300, help='Dim of PCA')
parser.add_argument('--feat_hidden1', type=int, default=100, help='Dim of DNN hidden 1-layer.')
parser.add_argument('--feat_hidden2', type=int, default=20, help='Dim of DNN hidden 2-layer.')
parser.add_argument('--gcn_hidden1', type=int, default=32, help='Dim of GCN hidden 1-layer.')
parser.add_argument('--gcn_hidden2', type=int, default=8, help='Dim of GCN hidden 2-layer.')
parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--using_dec', type=bool, default=True, help='Using DEC loss.')
parser.add_argument('--using_mask', type=bool, default=True, help='Using mask for multi-dataset.')
parser.add_argument('--feat_w', type=float, default=10, help='Weight of DNN loss.')
parser.add_argument('--gcn_w', type=float, default=0.1, help='Weight of GCN loss.')
parser.add_argument('--dec_kl_w', type=float, default=100, help='Weight of DEC loss.')
parser.add_argument('--gcn_lr', type=float, default=0.01, help='Initial GNN learning rate.')
parser.add_argument('--gcn_decay', type=float, default=0.01, help='Initial decay rate.')
parser.add_argument('--dec_cluster_n', type=int, default=8, help='DEC cluster number.')
parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')
# ______________ Eval clustering Setting _________
parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.') 

params = parser.parse_args()
params.device = device

# ################ Path setting
data_root = './spatial_datasets/DLPFC/'
# '151507', '151508', '151509', '151510', '151669', '151670',
# '151671', '151672', '151673', '151674', '151675', '151676'
# Batch correction list
proj_list = ['151507', '151672', '151673']
save_root = './SEDR_result/UBC/'

# ################ Combining dataset
for proj_idx in range(len(proj_list)):
    adata_h5_tmp = load_ST_file(file_fold=os.path.join(data_root, proj_list[proj_idx]))
    adata_h5_tmp.obs['batch_label'] = np.ones(adata_h5_tmp.shape[0]) * proj_idx
    graph_dict_tmp = graph_construction(adata_h5_tmp.obsm['spatial'], adata_h5_tmp.shape[0], params)

    # ########## Load layer_guess label, if have
    df_label = pd.read_csv(os.path.join(data_root, proj_list[proj_idx], "metadata.tsv"), sep='\t')
    adata_h5_tmp.obs['layer_guess'] = np.array(df_label['layer_guess'].to_list())

    if proj_idx == 0:
        adata_h5 = adata_h5_tmp
        graph_dict = graph_dict_tmp
        proj_name = proj_list[proj_idx]
    else:
        var_names = adata_h5.var_names.intersection(adata_h5_tmp.var_names)
        adata_h5 = adata_h5[:, var_names]
        adata_h5_tmp = adata_h5_tmp[:, var_names]
        adata_h5 = adata_h5.concatenate(adata_h5_tmp)
        graph_dict = combine_graph_dict(graph_dict, graph_dict_tmp)
        proj_name = proj_name + '_' + proj_list[proj_idx]

print('Combined adata: (' + str(adata_h5.shape[0]) + ', ' + str(adata_h5.shape[1]) + ')')

params.cell_num = adata_h5.shape[0]
params.save_path = mk_dir(os.path.join(save_root, proj_name))

adata_X = adata_preprocess(adata_h5, min_cells=5, pca_n_comps=params.cell_feat_dim)

# ######################## Model training
sed_net = SEDR_Train(adata_X, graph_dict, params)
if params.using_dec:
    sed_net.train_with_dec()
else:
    sed_net.train_without_dec()
sed_feat, _, _, _ = sed_net.process()

np.savez(os.path.join(params.save_path, "SED_result.npz"), sed_feat=sed_feat, deep_Dim=params.feat_hidden2)

# ######################## SEDR analysis
adata_sed = anndata.AnnData(sed_feat)
adata_sed.obsm['spatial'] = adata_h5.obsm['spatial']
adata_sed.obs['batch_label'] = pd.Categorical(adata_h5.obs['batch_label'])
adata_sed.obs['layer_guess'] = pd.Categorical(adata_h5.obs['layer_guess'])
sc.pp.neighbors(adata_sed)
sc.tl.umap(adata_sed)
sc.tl.leiden(adata_sed, key_added="SEDR_leiden", resolution=params.eval_resolution)
sc.pl.umap(adata_sed, color=["layer_guess", "batch_label"], title=["layer_guess",  "Batch"], color_map="Tab10")
plt.savefig(os.path.join(params.save_path, "SEDR_plot.jpg"), bbox_inches='tight', dpi=150)



