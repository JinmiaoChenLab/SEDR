#
import os
import umap
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sknetwork.clustering import Louvain, KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering


def get_umap(node_embs):
    reducer = umap.UMAP(n_neighbors=20, metric='cosine', random_state=42)
    return reducer.fit_transform(node_embs)


def proc_clustering(feature, params):
    latent_z_adj = kneighbors_graph(feature, params.eval_graph_n, mode='connectivity', include_self=True)
    if params.eval_cluster_type == 'Louvain':
        print('==== Clustering by Louvain')
        cluster_handle = Louvain()
        result = cluster_handle.fit_transform(latent_z_adj)
    elif params.eval_cluster_type == 'KMeans':
        print('==== Clustering by KMeans')
        cluster_handle = KMeans(n_clusters=params.eval_cluster_n)
        result = cluster_handle.fit_transform(latent_z_adj)
    else:
        print('==== Clustering by Spectral Clustering')
        cluster_handle = SpectralClustering(n_clusters=params.eval_cluster_n, affinity='precomputed',
                                            n_init=100, assign_labels='discretize', random_state=0)
        result = cluster_handle.fit_predict(latent_z_adj)
    return result


def eval_clustering(feat_dict, save_path, params, label=None):
    # clustering
    sed_labels = proc_clustering(feat_dict['sed_feat'], params)
    df_result = pd.DataFrame(sed_labels, columns=['sed_labels'])

    if feat_dict['deep_labels'] is not None:
        df_result['deep_labels'] = proc_clustering(feat_dict['deep_feat'], params)
    if label is not None:
        df_result['layer_guess'] = label
    if feat_dict['gnn_feat'] is not None:
        df_result['gnn_labels'] = proc_clustering(feat_dict['gnn_feat'], params)

    df_result.to_csv(os.path.join(save_path, params.eval_cluster_type + "_k_" +
                                  str(params.eval_cluster_n) + "_result.tsv"), sep='\t', index=False)

    return sed_labels


def plot_umap(node_feature, latent_z, save_path, params, label=None, colormap='tab20'):
    print('==== Computing UMAP by node_feature')
    umap_org = get_umap(node_feature)
    print('==== Computing UMAP by latent_z')
    umap_node = get_umap(latent_z)

    if label is None:
        label = proc_clustering(latent_z, params)

    fig, ax_list = plt.subplots(1, 2, figsize=(6, 3))
    sns.scatterplot(x=umap_org[:, 0], y=umap_org[:, 1], hue=label, palette=colormap, s=4, ax=ax_list[0])
    sns.scatterplot(x=umap_node[:, 0], y=umap_node[:, 1], hue=label, palette=colormap, s=4, ax=ax_list[1])

    for m_idx in range(len(ax_list)):
        plt.setp(ax_list[m_idx].spines.values(), color='black')
        ax_list[m_idx].spines['left'].set_linewidth(0.2)
        ax_list[m_idx].spines['top'].set_linewidth(0.2)
        ax_list[m_idx].spines['bottom'].set_linewidth(0.2)
        ax_list[m_idx].spines['right'].set_linewidth(0.2)
        ax_list[m_idx].get_xaxis().set_visible(False)
        ax_list[m_idx].get_yaxis().set_visible(False)

    ax_list[0].get_legend().remove()
    ax_list[0].set_title("DNN", position=(0.5, 0.9), fontdict={'fontsize': 8})
    ax_list[1].legend(handletextpad=0.3, fontsize=4, frameon=False, bbox_to_anchor=(1.04, 1))
    ax_list[1].set_title("DEC", position=(0.5, 0.9), fontdict={'fontsize': 8})
    plt.savefig(os.path.join(save_path, "result_plot.jpg"), dpi=200, bbox_inches='tight')








