import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia_values = []
ch_scores  = []
db_scores  = []

def init_metrics(data, k_min=3, k_max=21, random_state=2):

	inertia_values.clear()
	ch_scores.clear()
	db_scores.clear()
	
	K = range(k_min, k_max)

	for k in K:
		km = KMeans(n_clusters=k, random_state=random_state)
		km.fit(data)
		inertia_values.append(km.inertia_)
		ch_scores.append(metrics.calinski_harabasz_score(data, km.labels_))
		db_scores.append(metrics.davies_bouldin_score(data, km.labels_))
		
		
def display_metrics(k_min=3, k_max=21):

	K = range(k_min, k_max)
	
	plt.figure(figsize=(12, 12))

	plt.subplot(311)
	plt.title("Inertia (within-cluster sum-of-squares)")
	plt.plot(K, inertia_values, 'ro:')
	plt.xticks(K);

	plt.subplot(312)
	plt.title("Calinski-Harabasz score")
	plt.plot(K, ch_scores, 'bo:')
	plt.xticks(K);

	plt.subplot(313)
	plt.title("Davies-Bouldin index")
	plt.plot(K, db_scores, 'go:')
	plt.xticks(K);

def get_cluster_counts(data, n, k):
    
    km = KMeans(n_clusters=k, random_state=42)
    km = km.fit(data)
    result = km.labels_
    
    result = pd.DataFrame(result, columns=['clusters'])
    result = result.groupby('clusters').size()
    result = result.reset_index(name='count')
    
    ret = np.zeros(n-1)
    ret[:k] = result.sort_values(by='count', ascending=False)['count'].to_numpy()
    return ret


def get_orig_cluster_centers(data, columns, k, pca, scaler):

    km = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=42)
    km = km.fit(data)
    pca_inverse = pca.inverse_transform(km.cluster_centers_)
    scaler_inverse = scaler.inverse_transform(pca_inverse)

    centroid_df = pd.DataFrame(scaler_inverse, columns=columns)
    
    ccounts = pd.DataFrame(km.labels_, columns=['clusters'])
    ccounts = ccounts.groupby('clusters').size()
    ccounts = ccounts.reset_index(name='count')
    centroid_df['count'] = ccounts['count']

    gender = centroid_df[['female', 'male', 'other']].idxmax(1)
    centroid_df['gender'] = gender
    centroid_df.drop(columns=['female', 'male', 'other'], inplace=True)

    offer_type = centroid_df[['bogo', 'disc']].round().idxmax(1)
    centroid_df['offer_type'] = offer_type
    centroid_df.drop(columns=['bogo', 'disc'], inplace=True)

    centroid_df['web'] = centroid_df['web'].abs().round()
    centroid_df['mobile'] = centroid_df['mobile'].abs().round()
    centroid_df['social'] = centroid_df['social'].abs().round()

    return centroid_df