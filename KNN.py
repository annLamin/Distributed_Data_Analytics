import numpy as np

dataset_size = 10000
num_features = 10
k = 7
num_classes = 3

synthetic_dataset = np.random.rand(dataset_size, num_features)
euclidean_distances = np.zeros((dataset_size, 1))
query_datapoint = np.random.rand(1, num_features)

synthetic_labels = np.random.randint(0, num_classes, size=(dataset_size, 1))

synthetic_dataset = np.append(synthetic_dataset, synthetic_labels, axis=1)

for i in range(dataset_size) :
    current_distance = np.linalg.norm(query_datapoint - synthetic_dataset[i,:-1])
    euclidean_distances[i, :] = current_distance
# Sort the distances list to get the labels of the minimum k distances
sorted_indices = np.argsort(euclidean_distances, axis=0).flatten()
top_k_instances = synthetic_dataset[sorted_indices[:k]]
top_k_labels = top_k_instances[:, -1]
print(f"the class of the query instance is {np.bincount(top_k_labels.astype(int)).argmax()}")
