import faiss
import numpy as np
import os
print(faiss.omp_get_max_threads())
print(faiss.get_compile_options())
faiss.omp_set_num_threads(1)
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
os.environ['OMP_NUM_THREADS'] = "1"
os.environ['VECLIB_MAXIMUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"
print(faiss.omp_get_max_threads())


if __name__ == "__main__":
    num_dimensions = 1024
    num_vectors = 262144
    num_centroids = 1024
    threads = 1
    niter = 1

    data = np.fromfile("data_random.bin", dtype=np.float32).reshape(num_vectors, num_dimensions)
    data = data.astype(np.float32)
    print(data.shape)
    km = faiss.Kmeans(num_dimensions, num_centroids, niter=niter, verbose=True)
    km.train(data)
