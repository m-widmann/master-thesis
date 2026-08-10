from tslearn.clustering import TimeSeriesKMeans
import numpy as np

def distance_weights(dists):
    """
    Each trial gets assigned a weight based on its similarity to the centroid of the best cluster.
    Similarity is computed using dtw distance, and converted to a weight using an exponential kernel.
    """
    sigma = np.median(dists)
    weights = np.exp(-dists / (sigma + 1e-12))
    return weights


def dtw_clustering_to_get_response_trials(
        arr,
        window=None,
        n_clusters=2,
        metric='dtw',
        metric_parameters=None,
        max_iter_barycenter=50,
        random_state=0,
        reliability_threshold=0.2,
        n_init=5):
    """
    input: array in shape (trials, timestamps)
    ------------------------------------------------
    output: array in shape (timestamps[window],)
            (array is the averaged centroid of the selected cluster)


    Cluster the traces of one trial to filter out trials where there was no response from the cell.
    The best cluster gets selected by comparing absolute peak - std for each cluster.
    Then the centroid/barycenter (idealized signal) is returned.
    """
    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric=metric,
        metric_params=metric_parameters,
        max_iter_barycenter=max_iter_barycenter,
        random_state=random_state,
        n_init=n_init
    )
    if window is not None:
        arr = arr[:,window[0]:window[1]]
    labels = model.fit_predict(arr)
    centroids = model.cluster_centers_.squeeze() # get into 2d
    #pick best cluster
    peak_to_peak = np.quantile(centroids,.95, axis=1) - np.quantile(centroids,.05, axis=1) #gets both activation and inhibition
    step_variablity = np.nanmedian(np.abs(centroids[:,1:]-centroids[:,:-1]), axis=1)
    score = (peak_to_peak * (1/step_variablity)) #select high changes in amplitude but prefer low variability across the signal (which is more likely to be a real response rather than noise)
    best_cluster = score.argmax()
    reliability = np.unique(labels, return_counts=True)[1][best_cluster] / arr.shape[0]

    if reliability <= reliability_threshold:
        return np.concat([np.nanmedian(arr, axis=0), np.array([reliability])])
        # if the best cluster is less than threshold fraction of the trials, return the median of all trials instead of the centroid
        # Still return reliability to be able to filter out these cases later if desired.

    dist_matrix = model.transform(arr)
    dists = dist_matrix[:, best_cluster]

    # now still use all trials but weight them by how close they are to the centroid and how well their peaks match the centroid peaks
    # but the signal will never be higher than the initial centroid that was chosen
    weights = distance_weights(dists, labels, best_cluster)[:, None]
    weighted_sum = np.sum(weights * arr, axis=0)
    normalization = np.sum(weights, axis=0) + 1e-12
    final_signal = weighted_sum / normalization

    return np.concat([final_signal,np.array([reliability])])

