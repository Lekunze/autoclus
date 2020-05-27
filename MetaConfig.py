import sys
sys.path.append('../')
from metafeatures import Meta

import pandas as pd 
import numpy as np
from scipy.spatial.distance import cdist

class Config:

    def __init__(self, file, meta_type):
        self.file = file
        self.meta_type = meta_type

    def extract_metafeature(self):
        mf = Meta(self.file)
        return mf.extract_metafeatures(self.file, self.meta_type)

    def stock(self, index):
        if index == 'iindex':
            return 'i_index'
        elif index == 'sdbw':
            return 's_dbw'
        else:
            return index

    def format_cvi(self, index):
        # Evaluation Labels
        eval_labels = {"baker_hubert_gamma": -1, "banfeld_raferty": -1, "davies_bouldin": -1, "dunns_index": 1,
                       "mcclain_rao": -1, "pbm_index": 1, "ratkowsky_lance": 1, "ray_turi": -1, "scott_symons": -1,
                       "wemmert_gancarski": 1, "xie_beni": -1, "c_index": -1, "g_plus_index": -1, "i_index": 1,
                       "modified_hubert_t": 1, "point_biserial": 1, "s_dbw": -1, "silhouette": 1, "tau_index": 1,
                       "IIndex": 1, "SDBW": -1, "calinski_harabasz_score": 1}

        # Format CVI recommendation result
        index = index.strip('[')
        index = index.strip(']')
        index = index.replace("'", "")
        index = index.split(',')

        # For 3 CVIs get measure, return required format

        cvi1 = index[0].lower().strip()
        cvi2 = index[1].lower().strip()
        cvi3 = index[2].lower().strip()

        print(cvi1)
        print(cvi2)
        print(cvi3)

        # Final check for SDBW and IIndex

        cvi1 = self.stock(cvi1)
        cvi2 = self.stock(cvi2)
        cvi3 = self.stock(cvi3)

        return [[cvi1, eval_labels[cvi1]], [cvi2, eval_labels[cvi2]], [cvi3, eval_labels[cvi3]]]

        
        
    def search(self):

        # Get other meta-features from knowledge-base & their CVI combinations
        df_meta_db = pd.read_csv("metafeatures.csv")

        ds_name = self.file.split(".")[1].split("/")[2]
        df_meta_db = df_meta_db[df_meta_db.dataset != ds_name]

        df_meta_instance = self.extract_metafeature()
        df_meta_db = df_meta_db.append(df_meta_instance)

        df_algorithms = pd.read_csv("configs.csv")

        # Compute Euclidean distance between instance and other metafeatures
        df_meta_db_val = df_meta_db.loc[:, df_meta_db.columns != 'dataset']
        distance_matrix = cdist(df_meta_db_val, df_meta_db_val, metric = 'euclidean')
        
        instance_index = len(df_meta_db) - 1
        distances = np.trim_zeros(distance_matrix[instance_index])
        distances_sm = np.sort(distances)[0]

        index = np.where(distances == distances_sm)
        ds = str(df_meta_db.iloc[index].dataset.values[0])

        configs = df_algorithms.loc[df_algorithms['dataset'] == (ds)]['configuration']
        cvi = df_algorithms.loc[df_algorithms['dataset'] == (ds)]['cvi']
        algorithm = df_algorithms.loc[df_algorithms['dataset'] == (ds)]['algorithm']

        # Format configurations
        configs = configs.tolist()[0]
        cvi = cvi.tolist()[0]
        algorithm = algorithm.tolist()[0]

        return configs, cvi, algorithm
       

# config = Config("./Datasets/processed/gaussians1.csv", "distance")
# cf, cv, ag = config.search()
#
# print(cf)
