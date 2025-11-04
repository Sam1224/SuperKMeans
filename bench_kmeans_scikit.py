from sklearn.cluster import KMeans
import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = "1"

if __name__ == "__main__":
    num_dimensions = 1024
    num_vectors = 262144
    num_centroids = 4096
    threads = 1

    data = np.fromfile("data_random.bin", dtype=np.float32).reshape(num_vectors, num_dimensions)
    data = data.astype(np.float32)
    print(data.shape)
    km = KMeans(
        n_clusters=num_centroids,
        init='random',
        n_init=1,
        max_iter=25,
        verbose=1,
        random_state=1
    )
    km.fit(data)
