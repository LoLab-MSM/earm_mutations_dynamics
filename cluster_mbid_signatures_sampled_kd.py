from tropical import clustering
import pickle
import numpy as np

with open('signatures_sampled_kd_bcl2.pickle', 'rb') as handle:
    all_signatures = pickle.load(handle)

cpus = 30
rxn_type = 'consumption'

sp37_signatures = all_signatures[37][rxn_type]
clus = clustering.ClusterSequences(seqdata=sp37_signatures, unique_sequences=False, truncate_seq=50)
# diss = np.load('sampled_kd_37_diss.npy')
# clus.diss = diss
clus.diss_matrix(n_jobs=cpus)
np.save('sampled_kd_37_diss.npy', clus.diss)

sil_df = clus.silhouette_score_spectral_range(cluster_range=range(2, 20), n_jobs=cpus, random_state=1234)
print (sil_df)
n_clus, best_silh = sil_df.loc[sil_df['cluster_silhouette'].idxmax()]
n_clus = int(n_clus)
# n_clus = 7

clus.spectral_clustering(n_clusters=n_clus, n_jobs=cpus, random_state=1234)
np.save('clus_labels_sampled_kd.npy', clus.labels)
#b = clustering.PlotSequences(clus)
#b.modal_plot(title='Mitochondrial Bid')
#b.all_trajectories_plot(title='Mitochondrial Bid')

