import numpy as np

def process_symmetric_entries(matrix):
    transpose = matrix.T

    # Find indices where matrix and transpose are not equal
    indices = np.where(matrix != transpose)

    # For these indices, if one of the entries is zero, set it to the non-zero entry. If both are non-zero, calculate the average.
    matrix[indices] = np.where((matrix[indices] == 0), transpose[indices], 
                                     np.where((transpose[indices] == 0), matrix[indices], 
                                              (matrix[indices] + transpose[indices]) / 2))
    
    return matrix

def dfs(adj_matrix, start_vertex, visited):
    visited[start_vertex] = True
    stack = [start_vertex]
    subgraph_nodes = [start_vertex]

    while stack:
        vertex = stack.pop()
        for neighbor, connected in enumerate(adj_matrix[vertex]):
            if connected and not visited[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)
                subgraph_nodes.append(neighbor)

    return subgraph_nodes

def find_subgraphs(adj_matrix):
    num_vertices = len(adj_matrix)
    visited = [False] * num_vertices
    subgraphs = []

    for vertex in range(num_vertices):
        if not visited[vertex]:
            subgraph_nodes = dfs(adj_matrix, vertex, visited)
            subgraphs.append(subgraph_nodes)

    return subgraphs

def nodes_to_remove(subgraphs, threshold):
    nodes_to_remove = []

    for subgraph in subgraphs:
        if len(subgraph) < threshold:
            nodes_to_remove.extend(subgraph)

    return list(set(nodes_to_remove))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))