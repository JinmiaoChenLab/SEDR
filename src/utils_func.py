#
import os
import scanpy as sc
import pandas as pd
from pathlib import Path
from scanpy.readwrite import read_visium
from scanpy._utils  import check_presence_download


def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path


def adata_preprocess(i_adata, min_cells=3, pca_n_comps=300):
    print('===== Preprocessing Data ')
    sc.pp.filter_genes(i_adata, min_cells=min_cells)
    adata_X = sc.pp.normalize_total(i_adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    adata_X = sc.pp.scale(adata_X)
    adata_X = sc.pp.pca(adata_X, n_comps=pca_n_comps)
    return adata_X


def load_ST_file(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True, file_Adj=None):
    adata_h5 = sc.read_visium(file_fold, load_images=load_images, count_file=count_file)
    adata_h5.var_names_make_unique()

    if load_images is False:
        if file_Adj is None:
            file_Adj = os.path.join(file_fold, "spatial/tissue_positions_list.csv")

        positions = pd.read_csv(file_Adj, header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_col_in_fullres',
            'pxl_row_in_fullres',
        ]
        positions.index = positions['barcode']
        adata_h5.obs = adata_h5.obs.join(positions, how="left")
        adata_h5.obsm['spatial'] = adata_h5.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        adata_h5.obs.drop(columns=['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'], inplace=True)

    print('adata: (' + str(adata_h5.shape[0]) + ', ' + str(adata_h5.shape[1]) + ')')
    return adata_h5


# from scanpy
def _download_visium_dataset(
    sample_id: str,
    spaceranger_version: str,
    base_dir='./data/',
):
    import tarfile

    url_prefix = f'https://cf.10xgenomics.com/samples/spatial-exp/{spaceranger_version}/{sample_id}/'

    sample_dir = Path(mk_dir(os.path.join(base_dir, sample_id)))

    # Download spatial data
    tar_filename = f"{sample_id}_spatial.tar.gz"
    tar_pth = Path(os.path.join(sample_dir, tar_filename))
    check_presence_download(filename=tar_pth, backup_url=url_prefix + tar_filename)
    with tarfile.open(tar_pth) as f:
        for el in f:
            if not (sample_dir / el.name).exists():
                f.extract(el, sample_dir)

    # Download counts
    check_presence_download(
        filename=sample_dir / "filtered_feature_bc_matrix.h5",
        backup_url=url_prefix + f"{sample_id}_filtered_feature_bc_matrix.h5",
    )


def load_visium_sge(sample_id='V1_Breast_Cancer_Block_A_Section_1', save_path='./data/'):
    if "V1_" in sample_id:
        spaceranger_version = "1.1.0"
    else:
        spaceranger_version = "1.2.0"
    _download_visium_dataset(sample_id, spaceranger_version, base_dir=save_path)
    adata = read_visium(os.path.join(save_path, sample_id))

    print('adata: (' + str(adata.shape[0]) + ', ' + str(adata.shape[1]) + ')')
    return adata
