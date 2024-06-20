import torch

from torch_geometric import EdgeIndex
from torch_geometric.utils import scatter
from sklearn.cluster import SpectralClustering, KMeans
from uspec import uspec
import numpy as np

def sgc(data, K=2):
    edge_index = data.edge_index
    row, col = data.edge_index
    num_nodes = data.num_nodes
    edge_weight = torch.ones(data.num_edges, device=edge_index.device)

    deg = scatter(edge_weight, row, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    edge_index = EdgeIndex(edge_index, sparse_size=(num_nodes, num_nodes)).to(edge_index.device)
    edge_index, perm = edge_index.sort_by('col')
    edge_weight = edge_weight[perm]

    x = data.x
    for k in range(K):
        x = edge_index.matmul(x, edge_weight, transpose=True)

    data.x = x
    return data

def spectral_cluster(x, k=3):

    clustering = SpectralClustering(n_clusters=k, 
                                assign_labels='kmeans',
                                affinity='nearest_neighbors',  # 'nearest_neighbors' or 'rbf
                                random_state=0).fit(x.numpy())  # Adjust the number of clusters as needed
    return clustering.labels_

def rbf_kernel(X, Y=None, gamma=None):
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[1]  # Default to 1 / number of features
    K = torch.cdist(X, Y).pow(2)
    K = torch.exp(-gamma * K)
    return K

def nystrom_method(X, subset_indices, gamma):
    subset_data = X[subset_indices]

    # Compute the affinity matrix for the subset
    W = rbf_kernel(subset_data, gamma=gamma)

    # Compute the cross-affinity matrix between all points and the subset
    C = rbf_kernel(X, subset_data, gamma=gamma)

    # Approximate the full affinity matrix using Nyström method
    W_inv = torch.linalg.pinv(W)
    K_approx = torch.matmul(C, torch.matmul(W_inv, C.T))

    return K_approx

def batch_spectral_clustering(data, num_clusters, subset_size, batch_size, gamma=1.0):
    # Step 1: Random sampling
    indices = np.random.choice(data.shape[0], subset_size, replace=False)
    subset_data = data[indices].cpu()

    # Step 2: Compute the approximated similarity matrix using the Nyström method
    K_approx = nystrom_method(data, indices, gamma).cpu().numpy()

    # Step 3: Spectral clustering on the subset
    spectral = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', assign_labels='kmeans')
    subset_labels = spectral.fit_predict(K_approx[indices][:, indices])

    # Compute cluster centroids from the subset clustering
    kmeans = KMeans(n_clusters=num_clusters)
    subset_centroids = kmeans.fit(subset_data).cluster_centers_

    # Step 4: Assign remaining data points in batches
    remaining_indices = np.setdiff1d(np.arange(data.shape[0]), indices)
    remaining_data = data[remaining_indices]

    # Initialize an array for labels of the remaining data
    remaining_labels = np.empty(remaining_data.shape[0], dtype=int)

    # Process remaining data in batches
    for start in range(0, remaining_data.shape[0], batch_size):
        end = min(start + batch_size, remaining_data.shape[0])
        batch_data = remaining_data[start:end]

        # Compute similarity between the batch and cluster centroids
        centroids_torch = torch.from_numpy(subset_centroids).float().to(batch_data.device)
        pairwise_distances_batch = torch.cdist(batch_data, centroids_torch, p=2)
        similarity_to_centroids = torch.exp(-gamma * pairwise_distances_batch**2).cpu().numpy()

        # Assign each point in the batch to the nearest centroid
        batch_labels = np.argmax(similarity_to_centroids, axis=1)

        # Store batch labels
        remaining_labels[start:end] = batch_labels

    # Combine labels
    labels = np.empty(data.shape[0], dtype=int)
    labels[indices] = subset_labels
    labels[remaining_indices] = remaining_labels

    return labels


def h2gc(data, K=2, num_clusters=3, sparse=False):
    # cluster_array = batch_spectral_clustering(data.x, num_clusters=num_clusters, subset_size=1024, batch_size=8192)
    cluster_array = uspec(data.x.numpy(), Ks=[num_clusters]).squeeze()
    cluster_assignment = torch.from_numpy(cluster_array).to(data.x.device)

    edge_index = data.edge_index
    row, col = data.edge_index
    num_nodes = data.num_nodes
    edge_weight = torch.ones(data.num_edges, device=edge_index.device)

    deg = scatter(edge_weight, row, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    del deg, deg_inv_sqrt, row, col
    torch.cuda.empty_cache()

    edge_index = EdgeIndex(edge_index, sparse_size=(num_nodes, num_nodes)).to(edge_index.device)
    edge_index, perm = edge_index.sort_by('col')
    edge_weight = edge_weight[perm]

    print(f"Edge index shape: {edge_index.shape}")

    same_label_edge = cluster_assignment[edge_index[0]] == cluster_assignment[edge_index[1]]
    self_edge = edge_index[0] == edge_index[1] 
    homo_edge_mask = same_label_edge
    hetero_edge_mask = ~same_label_edge | self_edge

    homo_edge_index = edge_index[:, homo_edge_mask]
    homo_edge_weight = edge_weight[homo_edge_mask]
    homo_edge_index, perm = homo_edge_index.sort_by('col')
    homo_edge_weight = homo_edge_weight[perm]
    print(f"Homo Edge Index: Shape {homo_edge_index.shape}")

    hetero_edge_index = edge_index[:, hetero_edge_mask]
    hetero_edge_weight = edge_weight[hetero_edge_mask]
    hetero_edge_index, perm = hetero_edge_index.sort_by('col')
    hetero_edge_weight = hetero_edge_weight[perm]
    print(f"Hetero Edge Index: Shape {hetero_edge_index.shape}")

    print(f'Calculating K=1 data')
    data['x1'] = homo_edge_index.matmul(data.x, homo_edge_weight, transpose=True)
    data['x2'] = hetero_edge_index.matmul(data.x, hetero_edge_weight, transpose=True)

    if K == 1:
        return data
    print(f'Calculating K=2 data')

    data['x3'] = homo_edge_index.matmul(data['x1'], homo_edge_weight, transpose=True)
    data['x4'] = homo_edge_index.matmul(data['x2'], homo_edge_weight, transpose=True) + hetero_edge_index.matmul(data['x1'], hetero_edge_weight, transpose=True)

    batch_size = 2**26 //data.x.size(1)
    length = hetero_edge_index.size(1)
    # if batch_size > length:
    #     h2_edge_index, h2_edge_weight = hetero_edge_index.matmul(hetero_edge_index, hetero_edge_weight, hetero_edge_weight)
    #     h2_edge_index, perm = h2_edge_index.sort_by('col')
    #     h2_edge_weight = h2_edge_weight[perm]

    #     same_label_edge = cluster_assignment[h2_edge_index[0]] == cluster_assignment[h2_edge_index[1]]
    #     self_edge = h2_edge_index[0] == h2_edge_index[1]
    #     homo_h2_edge_index = h2_edge_index[:, same_label_edge].to('cuda')
    #     homo_h2_edge_weight = h2_edge_weight[same_label_edge].to('cuda')
    #     homo_h2_edge_index, perm = homo_h2_edge_index.sort_by('col')
    #     homo_h2_edge_weight = homo_h2_edge_weight[perm]

    #     data['x3'] += homo_h2_edge_index.matmul(data.x, homo_h2_edge_weight, transpose=True)

    #     hetero_h2_edge_index = h2_edge_index[:, ~same_label_edge | self_edge].to('cuda')
    #     hetero_h2_edge_weight = h2_edge_weight[~same_label_edge | self_edge].to('cuda')
    #     hetero_h2_edge_index, perm = hetero_h2_edge_index.sort_by('col')
    #     hetero_h2_edge_weight = hetero_h2_edge_weight[perm]
        
    #     data['x4'] += hetero_h2_edge_index.matmul(data.x, hetero_h2_edge_weight, transpose=True)

    #     return data
    
    hetero_edge_index = hetero_edge_index.to('cuda')
    hetero_edge_weight = hetero_edge_weight.to('cuda')
    cluster_assignment = cluster_assignment.to('cuda')
    data.x = data.x.to('cuda')
    for i in range(0, length, batch_size):
        print(f'Calculating {i} of {length}')
        if (hetero_edge_index.size(1) - batch_size) > i:
            curr_edge_index = hetero_edge_index[:, i:i+batch_size]
            curr_edge_weight = hetero_edge_weight[i:i+batch_size]
        else:
            curr_edge_index = hetero_edge_index[:, i:]
            curr_edge_weight = hetero_edge_weight[i:]
        h2_edge_index, h2_edge_weight = hetero_edge_index.matmul(curr_edge_index, hetero_edge_weight, curr_edge_weight)
        h2_edge_index, perm = h2_edge_index.sort_by('col')
        h2_edge_weight = h2_edge_weight[perm]

        same_label_edge = cluster_assignment[h2_edge_index[0]] == cluster_assignment[h2_edge_index[1]]
        self_edge = h2_edge_index[0] == h2_edge_index[1]
        homo_h2_edge_index = h2_edge_index[:, same_label_edge].to('cuda')
        homo_h2_edge_weight = h2_edge_weight[same_label_edge].to('cuda')
        homo_h2_edge_index, perm = homo_h2_edge_index.sort_by('col')
        homo_h2_edge_weight = homo_h2_edge_weight[perm]

        data['x3'] += homo_h2_edge_index.matmul(data.x, homo_h2_edge_weight, transpose=True).cpu()

        hetero_h2_edge_index = h2_edge_index[:, ~same_label_edge | self_edge].to('cuda')
        hetero_h2_edge_weight = h2_edge_weight[~same_label_edge | self_edge].to('cuda')
        hetero_h2_edge_index, perm = hetero_h2_edge_index.sort_by('col')
        hetero_h2_edge_weight = hetero_h2_edge_weight[perm]
        
        data['x4'] += hetero_h2_edge_index.matmul(data.x, hetero_h2_edge_weight, transpose=True).cpu()

        print(torch.cuda.max_memory_allocated()*2**(-20))
        torch.cuda.empty_cache()

    return data

def h2gc_sparse(data, K=2, num_clusters=3):
    cluster_array = uspec(data.x.numpy(), Ks=[num_clusters]).squeeze()
    cluster_assignment = torch.from_numpy(cluster_array).to(data.x.device)

    edge_index = data.edge_index
    row, col = data.edge_index
    num_nodes = data.num_nodes
    edge_weight = torch.ones(data.num_edges, device=edge_index.device)

    deg = scatter(edge_weight, row, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    edge_index = EdgeIndex(edge_index, sparse_size=(num_nodes, num_nodes)).to(edge_index.device)
    edge_index, perm = edge_index.sort_by('col')
    edge_weight = edge_weight[perm]
    
    original_edge_weight = edge_weight.clone()
    original_edge_index = edge_index.clone()

    labels = cluster_assignment
    for k in range(K):
        print(f"Starting iteration {k+1}")
        print(f"Edge index shape: {edge_index.shape}")
        
        same_label_edge = labels[edge_index[0]] == labels[edge_index[1]]
        self_edge = edge_index[0] == edge_index[1]
        homo_edge_mask = same_label_edge & ~self_edge
        hetero_edge_mask = ~same_label_edge & ~self_edge

        homo_edge_index = edge_index[:, homo_edge_mask]
        homo_edge_weight = edge_weight[homo_edge_mask]
        homo_edge_index, perm = homo_edge_index.sort_by('col')
        homo_edge_weight = homo_edge_weight[perm]
        print(f"Homo Edge Index: Shape {homo_edge_index.shape}")

        hetero_edge_index = edge_index[:, hetero_edge_mask]
        hetero_edge_weight = edge_weight[hetero_edge_mask]
        hetero_edge_index, perm = hetero_edge_index.sort_by('col')
        hetero_edge_weight = hetero_edge_weight[perm]
        print(f"Hetero Edge Index: Shape {hetero_edge_index.shape}")

        data[f'x{k*2+1}'] = homo_edge_index.matmul(data.x, homo_edge_weight, transpose=True)
        data[f'x{k*2+2}'] = hetero_edge_index.matmul(data.x, hetero_edge_weight, transpose=True)
        print(f"Iteration {k+1} done")
        if k != K-1:
            edge_index, edge_weight = edge_index.matmul(original_edge_index, edge_weight, original_edge_weight)
            edge_index, perm = edge_index.sort_by('col')
            edge_weight = edge_weight[perm]
    return data

def hhsign(data, K=2, num_clusters=3):
    edge_index = data.edge_index
    row, col = data.edge_index
    num_nodes = data.num_nodes
    edge_weight = torch.ones(data.num_edges, device=edge_index.device)

    deg = scatter(edge_weight, row, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    edge_index = EdgeIndex(edge_index, sparse_size=(num_nodes, num_nodes)).to(edge_index.device)
    edge_index, perm = edge_index.sort_by('col')
    edge_weight = edge_weight[perm]
    
    original_edge_weight = edge_weight.clone()
    original_edge_index = edge_index.clone()

    labels = data.y
    val_mask = data.val_mask
    test_mask = data.test_mask
    for k in range(K):
        print(f"Starting iteration {k+1}")
        print(f"Edge index shape: {edge_index.shape}")
        
        same_label_edge = labels[edge_index[0]] == labels[edge_index[1]]
        is_val_edge = val_mask[edge_index[0]] | val_mask[edge_index[1]]
        is_test_edge = test_mask[edge_index[0]] | test_mask[edge_index[1]]
        self_edge = edge_index[0] == edge_index[1]
        homo_edge_mask = same_label_edge & ~self_edge
        hetero_edge_mask = ~same_label_edge & ~self_edge

        homo_edge_index = edge_index[:, homo_edge_mask]
        homo_edge_weight = edge_weight[homo_edge_mask]
        homo_edge_index, perm = homo_edge_index.sort_by('col')
        homo_edge_weight = homo_edge_weight[perm]
        print(f"Homo Edge Index: Shape {homo_edge_index.shape}")

        hetero_edge_index = edge_index[:, hetero_edge_mask]
        hetero_edge_weight = edge_weight[hetero_edge_mask]
        hetero_edge_index, perm = hetero_edge_index.sort_by('col')
        hetero_edge_weight = hetero_edge_weight[perm]
        print(f"Hetero Edge Index: Shape {hetero_edge_index.shape}")

        data[f'x{k*2+1}'] = homo_edge_index.matmul(data.x, homo_edge_weight, transpose=True)
        data[f'x{k*2+2}'] = hetero_edge_index.matmul(data.x, hetero_edge_weight, transpose=True)
        print(f"Iteration {k+1} done")
        if k != K-1:
            edge_index, edge_weight = edge_index.matmul(original_edge_index, edge_weight, original_edge_weight)
            edge_index, perm = edge_index.sort_by('col')
            edge_weight = edge_weight[perm]
    return data
