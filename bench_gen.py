import sklearn.datasets
import numpy as np

if __name__ == "__main__":
    num_dimensions = 1024
    num_vectors = 262144
    num_centroids = 4096

    print(f'Bench gen: \n- D={num_dimensions}\n- num_centroids={num_centroids}\n- dataset=RANDOM')
    data, _ = sklearn.datasets.make_blobs(n_samples=num_vectors, n_features=num_dimensions, centers=num_centroids, random_state=1)
    data = data.astype(np.float32)

    data.tofile("data_random.bin")
