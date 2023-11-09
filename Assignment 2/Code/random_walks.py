# %% [markdown]
# **Investigating Graph Node Embeddings**
# 
# Gian Favero | ECSE 556 | December 1st, 2023

# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from helpers import *

# %% [markdown]
# First we load the .edge file associated with the “HumanNet Co-Expression of Human Genes (hn_HS_CX) network. The file will be cleaned up in such a way that only the relevant columns and values are kept.

# %%
edges_df = pd.read_csv('9606.hn_HS_CX.edge', sep='\t', header=None)
edges_df = edges_df.iloc[:, :3]
edges_df.columns = ['Node 1', 'Node 2', 'Weight']
print(edges_df.head())

# %% [markdown]
# Now we have to start forming the adjacency matrix that represents the network. We can get a set of every node in the network and then augment the dataset to ensure the network is undirected.

# %%
# Get set of all nodes
nodes = set(edges_df['Node 1'])
nodes = nodes.union(set(edges_df['Node 2']))

# Convert nodes to indices in edges_df
nodes_dict = dict(zip(nodes, range(len(nodes))))
edge_id_df = edges_df.copy()
edge_id_df['Node 1'] = edge_id_df['Node 1'].map(nodes_dict)
edge_id_df['Node 2'] = edge_id_df['Node 2'].map(nodes_dict)

# Initialize adjacency matrix
adj_mat = np.zeros((len(nodes), len(nodes)))

# Fill adjacency matrix
for i in tqdm(range(len(edge_id_df))):
    row = edge_id_df.iloc[i]
    adj_mat[int(row['Node 1']), int(row['Node 2'])] = row['Weight']

# If there are any self-loops, remove them
np.fill_diagonal(adj_mat, 0)

# %%
# Adjust adjacency matrix to be symmetric, undirected
adj_mat = process_symmetric_entries(adj_mat)

# Find all subgraphs and get list of nodes that belong to subgraphs with less than 5 nodes
subgraphs = find_subgraphs(adj_mat)
nodes_remove = nodes_to_remove(subgraphs, 5)

# Remove nodes from adjacency matrix
adj_mat = np.delete(adj_mat, nodes_remove, axis=0)
adj_mat = np.delete(adj_mat, nodes_remove, axis=1)

# Remove nodes from nodes list
nodes = list(nodes)
nodes = [nodes[i] for i in range(len(nodes)) if i not in nodes_remove]
nodes_dict = dict(zip(nodes, range(len(nodes))))

# Remove nodes from edges_df
edges_df = edges_df[~edges_df['Node 1'].isin(nodes_remove)]
edges_df = edges_df[~edges_df['Node 2'].isin(nodes_remove)]

edges_df['Node 1'] = edges_df['Node 1'].map(nodes_dict)
edges_df['Node 2'] = edges_df['Node 2'].map(nodes_dict)

# Normalize adjacency matrix by row
tr_mat = adj_mat / adj_mat.sum(axis=1, keepdims=True)

# %% [markdown]
# **RW No Restart**
# 
# Now, we initialize three random initial distribution vectors with only one non-zero entry. We run RW until convergence and output the final distributions to see if RW without restart preserves local information about the nodes.

# %%
# First random walk vector has only one entry of 1, rest are 0
rw_vec = np.zeros((tr_mat.shape[0], 1))
rw_vec1 = rw_vec
rw_vec1[0] = 1

rw_vec2 = rw_vec
rw_vec2[1500] = 1

rw_vec3 = rw_vec
rw_vec3[5000] = 1

vecs = [rw_vec1, rw_vec2, rw_vec3]

# Perform random walk until there is small change in vector
stat_dist_no_restart = []
eps = 1e-5
for vec in tqdm(vecs):
    while True:
        rw_vec_new = tr_mat @ vec
        if np.linalg.norm(rw_vec_new - vec) < eps:
            break
        vec = rw_vec_new
    stat_dist_no_restart.append(rw_vec_new)

# See if the vectors are equal
are_close = np.allclose(stat_dist_no_restart[0], stat_dist_no_restart[1]) and \
            np.allclose(stat_dist_no_restart[0], stat_dist_no_restart[2]) and \
            np.allclose(stat_dist_no_restart[1], stat_dist_no_restart[2])
print('Are the vectors close? {}'.format(are_close))

# Get cosine similarity between vectors
cos_sim = []
for i in range(len(stat_dist_no_restart)):
    for j in range(i+1, len(stat_dist_no_restart)):
        dot = np.dot(stat_dist_no_restart[i].T, stat_dist_no_restart[j])[0][0]
        norm1 = np.linalg.norm(stat_dist_no_restart[i])
        norm2 = np.linalg.norm(stat_dist_no_restart[j])
        cos_sim.append(dot / (norm1 * norm2))
print('Cosine similarity between vectors: {}'.format(cos_sim))

# Get Pearson correlation between vectors
pearson_corr = []
for i in range(len(stat_dist_no_restart)):
    for j in range(i+1, len(stat_dist_no_restart)):
        pearson_corr.append(pearsonr(stat_dist_no_restart[i].flatten(), stat_dist_no_restart[j].flatten())[0])
print('Pearson correlation between vectors: {}'.format(pearson_corr))

# Get Spearman correlation between vectors
spearman_corr = []
for i in range(len(stat_dist_no_restart)):
    for j in range(i+1, len(stat_dist_no_restart)):
        spearman_corr.append(spearmanr(stat_dist_no_restart[i].flatten(), stat_dist_no_restart[j].flatten())[0])
print('Spearman correlation between vectors: {}'.format(spearman_corr))

# %% [markdown]
# **RW with Restart**
# 
# Now, we do a similar procedure as before but this time there is a chance on every iteration that the walk restarts from the beginning of its walk.

# %%
# Assign the restart nodes
N1 = int(edges_df['Node 1'].iloc[0])
v_N1 = np.zeros((tr_mat.shape[0], 1))
v_N1[N1] = 1

N2 = int(edges_df['Node 2'].iloc[0])
v_N2 = np.zeros((tr_mat.shape[0], 1))
v_N2[N2] = 1

# %% [markdown]
# Starting from an initial distribution, q0, we can find the stationary distribution when N1 and N2 are the restart set at different values of p

# %%
# Make list of p
p = [0.2, 0.5, 0.8]

# Make random initial distribution
q = np.zeros((tr_mat.shape[0], 1))
q[0] = 1

# %%
import matplotlib.pyplot as plt

for index, node in enumerate([v_N1, v_N2]):
    eps = 1e-5
    vec = q
    stat_dist_restart = []

    for p_i in p:
        while True:
            rw_vec_new = p_i * tr_mat @ vec + (1 - p_i) * node
            if np.linalg.norm(rw_vec_new - vec) < eps:
                break
            vec = rw_vec_new
        stat_dist_restart.append(rw_vec_new)

    # Put stationary distributions into a dataframe
    stat_dist_restart_df = pd.DataFrame(np.hstack(stat_dist_restart))
    stat_dist_restart_df.insert(0, 'Node', nodes)
    stat_dist_restart_df.columns = ['Node', 'p = 0.2', 'p = 0.5', 'p = 0.8']

    # Save stationary distributions to csv
    stat_dist_restart_df.to_csv(f'stationary/stat_dist_p_{index}.csv', index=False)

    # Compare stationary distributions to restart node 
    # Get cosine similarity between stationary distributions and restart node
    cos_sim = []
    for i in range(len(stat_dist_restart)):
        dot = np.dot(stat_dist_restart[i].T, node)
        norm1 = np.linalg.norm(stat_dist_restart[i])
        norm2 = np.linalg.norm(node)
        cosine_similarity = dot / (norm1 * norm2)
        cos_sim.append(cosine_similarity.item())

    # Plot cosine similarity vs p
    plt.figure(figsize=(4, 3))
    plt.plot(p, cos_sim, '-o')
    plt.xlabel('p')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity vs p')
    plt.show()

# %% [markdown]
# We can also observe what happens when a different initial distribution is picked but p is held fixed 

# %%
# Enforce constant p
p_fixed = 0.2

# Make 3 random initial distributions
q1 = np.zeros((tr_mat.shape[0], 1))
q2 = np.zeros((tr_mat.shape[0], 1))
q3 = np.zeros((tr_mat.shape[0], 1))

q1[0] = 1
q2[1500] = 1
q3[5000] = 1

qs = [q1, q2, q3]

# %%
import matplotlib.pyplot as plt

for index, node in enumerate([v_N1, v_N2]):
    eps = 1e-5
    stat_dist_restart = []

    for q_i in qs:
        vec = q_i
        while True:
            rw_vec_new = p_fixed * tr_mat @ vec + (1 - p_fixed) * node
            if np.linalg.norm(rw_vec_new - vec) < eps:
                break
            vec = rw_vec_new
        stat_dist_restart.append(rw_vec_new)

    # Compare stationary distributions to restart node 
    # Get cosine similarity between stationary distributions and restart node
    cos_sim = []
    for i in range(len(stat_dist_restart)):
        dot = np.dot(stat_dist_restart[i].T, node)
        norm1 = np.linalg.norm(stat_dist_restart[i])
        norm2 = np.linalg.norm(node)
        cosine_similarity = dot / (norm1 * norm2)
        cos_sim.append(cosine_similarity.item())
    
    # Plot cosine similarity vs p
    plt.figure(figsize=(4, 3))
    plt.plot(p, cos_sim, '-o')
    plt.xlabel('p')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity vs p')
    plt.ylim(0.98, 1.02)
    plt.show()

# %% [markdown]
# Last, we can analyze the benefits of restart. Are the stationary distributions generated using N1 and N2 the same as they were with no restart?

# %%
# Enforce constant p
p_fixed = 0.2

# Enforce a random initial distribution
q_fixed = np.zeros((tr_mat.shape[0], 1))
q_fixed[0] = 1

# %%
eps = 1e-5
stat_dist_restart = []
vec = q_fixed

for index, node in enumerate([v_N1, v_N2]):
    while True:
        rw_vec_new = p_fixed * tr_mat @ vec + (1 - p_fixed) * node
        if np.linalg.norm(rw_vec_new - vec) < eps:
            break
        vec = rw_vec_new
    stat_dist_restart.append(rw_vec_new)

# Check if stationary distributions are equal
are_close = np.allclose(stat_dist_restart[0], stat_dist_restart[1])
print('Are the stationary distributions close? {}'.format(are_close))

# Get cosine similarity between stationary distributions
dot = np.dot(stat_dist_restart[0].T, stat_dist_restart[1])
norm1 = np.linalg.norm(stat_dist_restart[0])
norm2 = np.linalg.norm(stat_dist_restart[1])
cosine_similarity = dot / (norm1 * norm2)
print('Cosine similarity between stationary distributions: {}'.format(cosine_similarity[0][0]))

# Get cosine similarity between stationary distributions and restart node 1
for i in range(2):
    dot = np.dot(stat_dist_restart[i].T, v_N1)
    norm1 = np.linalg.norm(stat_dist_restart[i])
    norm2 = np.linalg.norm(v_N1)
    cosine_similarity = dot / (norm1 * norm2)
    print('Cosine similarity between stationary distribution {} and restart node 1: {}'.format(i, cosine_similarity[0][0]))

# Get cosine similarity between stationary distributions and restart node 2
for i in range(2):
    dot = np.dot(stat_dist_restart[i].T, v_N2)
    norm1 = np.linalg.norm(stat_dist_restart[i])
    norm2 = np.linalg.norm(v_N2)
    cosine_similarity = dot / (norm1 * norm2)
    print('Cosine similarity between stationary distribution {} and restart node 2: {}'.format(i, cosine_similarity[0][0]))

# %% [markdown]
# **Node2Vec**
# 
# We now get embeddings from a Node2Vec implementation

# %%
import networkx as nx
import node2vec

# Create graph from adjacency matrix
G = nx.from_numpy_matrix(adj_mat)

# Get ranges of p and q to try
node_p = [0.5, 2, 4]
node_q = [0.5, 2, 4]

# Create list of node2vec models
node2vec_models = []
for p in node_p:
    for q in node_q:
        walk = node2vec.Node2Vec(G, p=p, q=q, workers=10)
        model = walk.fit()
        node2vec_models.append(model)

# Save node2vec models
for index, model in enumerate(node2vec_models):
    model.save(f'node2vec/node2vec_model_{index}.bin')

# %%
from gensim.models import Word2Vec

# Load node2vec models
depth_first_model = Word2Vec.load('node2vec/node2vec_model_6.bin')
breadth_first_model = Word2Vec.load('node2vec/node2vec_model_2.bin')

# %%
# Get depth-first embeddings for first two nodes
N1_depth_e = depth_first_model.wv[N1]
N2_depth_e = depth_first_model.wv[N2]

# Get breadth-first embeddings for first two nodes
N1_breadth_e = breadth_first_model.wv[N1]
N2_breadth_e = breadth_first_model.wv[N2]

n1_embeddings = [N1_depth_e, N1_breadth_e]
n2_embeddings = [N2_depth_e, N2_breadth_e]

# Get Pearsons
n1_pearsons = []
n2_pearsons = []
for i in range(len(n1_embeddings)):
    for j in range(i + 1, len(n1_embeddings)):
        pearson = pearsonr(n1_embeddings[i], n1_embeddings[j])
        n1_pearsons.append(pearson[0])
        pearson = pearsonr(n2_embeddings[i], n2_embeddings[j])
        n2_pearsons.append(pearson[0])

# Get Spearmans
n1_spearmans = []
n2_spearmans = []
for i in range(len(n1_embeddings)):
    for j in range(i + 1, len(n1_embeddings)):
        spearman = spearmanr(n1_embeddings[i], n1_embeddings[j])
        n1_spearmans.append(spearman[0])
        spearman = spearmanr(n2_embeddings[i], n2_embeddings[j])
        n2_spearmans.append(spearman[0])

# Get cosine similarities
n1_cos_sim = []
n2_cos_sim = []
for i in range(len(n1_embeddings)):
    for j in range(i + 1, len(n1_embeddings)):
        dot = np.dot(n1_embeddings[i].T, n1_embeddings[j])
        norm1 = np.linalg.norm(n1_embeddings[i])
        norm2 = np.linalg.norm(n1_embeddings[j])
        cosine_similarity = dot / (norm1 * norm2)
        n1_cos_sim.append(cosine_similarity.item())
        dot = np.dot(n2_embeddings[i].T, n2_embeddings[j])
        norm1 = np.linalg.norm(n2_embeddings[i])
        norm2 = np.linalg.norm(n2_embeddings[j])
        cosine_similarity = dot / (norm1 * norm2)
        n2_cos_sim.append(cosine_similarity.item())

# Print results
print('Node 1')
print('Pearson: {}'.format(n1_pearsons))
print('Spearman: {}'.format(n1_spearmans))
print('Cosine Similarity: {}'.format(n1_cos_sim))

print('\nNode 2')
print('Pearson: {}'.format(n2_pearsons))
print('Spearman: {}'.format(n2_spearmans))
print('Cosine Similarity: {}'.format(n2_cos_sim))

# %% [markdown]
# Now, we train various node2vec models across ranges of p and q to analyze the correlations

# %%


# %%
''' Test code for adj matrix processing '''

'''# Define nodes and weights
node1 = ['A', 'B', 'C', 'D', 'B']
node2 = ['B', 'C', 'D', 'A', 'A']
weights = [1.0, 2.0, 3.0, 4.0, 5.0]

# Create dictionary
data = {'Node 1': node1, 'Node 2': node2, 'Weight': weights}

# Convert to DataFrame
test_df = pd.DataFrame(data)

print(test_df.head())

# Get set of all nodes
nodes = set(test_df['Node 1'])
nodes = nodes.union(set(test_df['Node 2']))

# Convert nodes to indices in edges_df
nodes_dict = dict(zip(nodes, range(len(nodes))))
test_df['Node 1'] = test_df['Node 1'].map(nodes_dict)
test_df['Node 2'] = test_df['Node 2'].map(nodes_dict)

print(test_df.head())

# Initialize adjacency matrix
test_adj = np.zeros((len(nodes), len(nodes)))

print(test_adj)

# Fill adjacency matrix
for i in range(len(test_df)):
    row = test_df.iloc[i]
    test_adj[int(row['Node 1']), int(row['Node 2'])] = row['Weight']

print(test_adj)

test_adj = process_symmetric_entries(test_adj)

print(test_adj)

# normalize adjacency matrix by row
test_adj = test_adj / test_adj.sum(axis=1, keepdims=True)

print(test_adj)'''


