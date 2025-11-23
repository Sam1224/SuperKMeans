import os
threads = "10"
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["BLIS_NUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads


from sklearn.cluster import KMeans
import numpy as np
import os
import time
import math
import fastkmeans
from fastkmeans import FastKMeans

if __name__ == "__main__":
    num_dimensions = 1024
    num_vectors = 262144
    num_centroids = 1024
    threads = 10
    n_iter = 25

    data = np.fromfile("data_mxbai.bin", dtype=np.float32).reshape(num_vectors, num_dimensions)
    data = data.astype(np.float32)
    print(data.shape)
    start = time.time()
    km = KMeans(
        n_clusters=num_centroids,
        init='random',
        n_init=1,
        max_iter=n_iter,
        verbose=1,
        random_state=42,
        copy_x=True
    )
    km.fit(data)
    end = time.time()
    print(f"ScikitLearn Took: {(end - start):.2f} s")

    start = time.time()
    kmeans = FastKMeans(
        d=num_dimensions,
        k=num_centroids,
        niter=n_iter,
        tol=-math.inf,
        device='cpu',
        seed=42,
        max_points_per_centroid=256,
        verbose=True,
        use_triton=False,
    )
    kmeans.train(data)
    end = time.time()
    print(f"FastKMeans Took: {(end - start):.2f} s")
