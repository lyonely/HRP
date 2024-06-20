import numpy as np
import torch
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def get_representatives_by_hybrid_selection(fea, p_size, distance, cnt_times=10):
    n = fea.shape[0]
    big_p_size = cnt_times * p_size
    if p_size > n:
        p_size = n
    if big_p_size > n:
        big_p_size = n

    big_rp_fea, _ = get_representatives_by_random_selection(fea, big_p_size)
    kmeans = KMeans(n_clusters=p_size, max_iter=10).fit(big_rp_fea)
    rp_fea = kmeans.cluster_centers_

    return rp_fea

def get_representatives_by_random_selection(fea, p_size):
    n = fea.shape[0]
    if p_size > n:
        p_size = n
    select_idxs = np.random.choice(n, p_size, replace=False)
    rp_fea = fea[select_idxs, :]
    return rp_fea, select_idxs

def uspec(fea, Ks, distance='euclidean', p=1000, Knn=5, maxTcutKmIters=100, cntTcutKmReps=3):
    N = fea.shape[0]
    if p > N:
        p = N

    # Convert torch tensor to numpy array if necessary
    if isinstance(fea, torch.Tensor):
        fea = fea.cpu().numpy()

    # Get p representatives by hybrid selection
    rp_fea = get_representatives_by_hybrid_selection(fea, p, distance)

    # Approx. KNN
    cnt_rep_cls = int(np.floor(np.sqrt(p)))
    kmeans = KMeans(n_clusters=cnt_rep_cls, max_iter=20).fit(rp_fea)
    rep_cls_label = kmeans.labels_
    rep_cls_centers = kmeans.cluster_centers_

    # Pre-compute the distance between N objects and the cnt_rep_cls rep-cluster centers
    center_dist = cdist(fea, rep_cls_centers, metric=distance)

    # Find the nearest rep-cluster for each object
    min_center_idxs = np.argmin(center_dist, axis=1)
    cnt_rep_cls = rep_cls_centers.shape[0]

    # Find the nearest representative in the nearest rep-cluster for each object
    nearest_rep_in_rp_fea_idx = np.zeros(N, dtype=int)
    for i in range(cnt_rep_cls):
        mask = (min_center_idxs == i)
        nearest_rep_in_rp_fea_idx[mask] = np.argmin(cdist(fea[mask], rp_fea[rep_cls_label == i], metric=distance), axis=1)
        tmp = np.where(rep_cls_label == i)[0]
        nearest_rep_in_rp_fea_idx[mask] = tmp[nearest_rep_in_rp_fea_idx[mask]]

    # For each object, compute its distance to the candidate neighborhood of its nearest representative
    neigh_size = 10 * Knn
    rp_fea_w = cdist(rp_fea, rp_fea, metric=distance)
    rp_fea_knn_idx = np.argsort(rp_fea_w, axis=1)[:, :neigh_size + 1]

    rp_fea_knn_dist = np.zeros((N, rp_fea_knn_idx.shape[1]))
    for i in range(p):
        mask = (nearest_rep_in_rp_fea_idx == i)
        rp_fea_knn_dist[mask] = cdist(fea[mask], rp_fea[rp_fea_knn_idx[i]], metric=distance)

    rp_fea_knn_idx_full = rp_fea_knn_idx[nearest_rep_in_rp_fea_idx]

    # Get the final KNN according to the candidate neighborhood
    knn_dist = np.zeros((N, Knn))
    knn_idx = np.zeros((N, Knn), dtype=int)
    for i in range(Knn):
        min_dist_idx = np.argmin(rp_fea_knn_dist, axis=1)
        knn_dist[:, i] = rp_fea_knn_dist[np.arange(N), min_dist_idx]
        knn_idx[:, i] = rp_fea_knn_idx_full[np.arange(N), min_dist_idx]
        rp_fea_knn_dist[np.arange(N), min_dist_idx] = np.inf

    # Compute the cross-affinity matrix B for the bipartite graph
    if distance == 'cosine':
        Gsdx = 1 - knn_dist
    else:
        knn_mean_diff = np.mean(knn_dist)
        Gsdx = np.exp(-knn_dist ** 2 / (2 * knn_mean_diff ** 2))

    Gsdx[Gsdx == 0] = np.finfo(float).eps
    Gidx = np.tile(np.arange(N).reshape(-1, 1), Knn)
    B = csr_matrix((Gsdx.ravel(), (Gidx.ravel(), knn_idx.ravel())), shape=(N, p))

    labels = np.zeros((N, len(Ks)))
    for i, K in enumerate(Ks):
        labels[:, i] = tcut_for_bipartite_graph(B, K, maxTcutKmIters, cntTcutKmReps)

    return labels

def tcut_for_bipartite_graph(B, Nseg, maxKmIters=100, cntReps=3):
    Nx, Ny = B.shape
    if Ny < Nseg:
        raise ValueError('Need more columns!')

    dx = np.array(B.sum(axis=1)).ravel()
    dx[dx == 0] = 1e-10  # To make 1./dx feasible
    Dx = csr_matrix((1. / dx, (np.arange(Nx), np.arange(Nx))), shape=(Nx, Nx))
    Wy = B.T @ Dx @ B

    # Normalized affinity matrix
    d = np.array(Wy.sum(axis=1)).ravel()
    D = csr_matrix((1. / np.sqrt(d), (np.arange(Ny), np.arange(Ny))), shape=(Ny, Ny))
    nWy = D @ Wy @ D
    nWy = (nWy + nWy.T) / 2

    # Compute eigenvectors
    evals, evecs = eigsh(nWy, k=Nseg, which='LM')
    idx = np.argsort(-evals)
    Ncut_evec = D @ evecs[:, idx[:Nseg]]

    # Compute the Ncut eigenvectors on the entire bipartite graph (transfer!)
    evec = Dx @ B @ Ncut_evec

    # Normalize each row to unit norm
    evec = evec / np.linalg.norm(evec, axis=1, keepdims=True)

    # K-means
    kmeans = KMeans(n_clusters=Nseg, max_iter=maxKmIters, n_init=cntReps).fit(evec)
    labels = kmeans.labels_

    return labels

# Example usage
if __name__ == "__main__":
    data = torch.randn(10000, 50)  # Example large dataset
    num_clusters = 10
    Ks = [num_clusters]
    labels = uspec(data, Ks)
    print("Cluster labels:", labels)
