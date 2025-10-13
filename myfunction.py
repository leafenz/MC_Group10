import numpy as np
import multiprocessing as mp
from multiprocessing import Lock



# ===============================================================
# GLOBAL SHARED VARIABLES
# ===============================================================
shared_space = None
shared_matrix = None


# ===============================================================
# Convert shared memory to NumPy array
# ===============================================================
def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj(), dtype=np.float64)


# ===============================================================
# Initializer (executed once per worker)
# ===============================================================
def init_sharedarray(shared_array, img_shape):
    global shared_space
    global shared_matrix
    shared_space = shared_array
    shared_matrix = tonumpyarray(shared_space).reshape(img_shape)


# ===============================================================
# Filter Class
# ===============================================================
class Filter_three_Dimension:
    def __init__(self, image, imagefilter):
        self.image = image.astype(np.float64)
        self.imagefilter = imagefilter.astype(np.float64)
        self.shape = image.shape
        self.lock = Lock()

    # Worker function (runs in parallel)
    def edge_filter_row(self, x):
        global shared_matrix
        image = self.image
        kernel = self.imagefilter
        (rows, cols, depth) = image.shape
        (kx, ky, kz) = kernel.shape
        cx, cy, cz = kx // 2, ky // 2, kz // 2

        frow = np.zeros((cols, depth))
        for y in range(cols):
            for z in range(depth):
                acc = 0.0
                for i in range(kx):
                    for j in range(ky):
                        for k in range(kz):
                            rr = min(max(x + i - cx, 0), rows - 1)
                            cc = min(max(y + j - cy, 0), cols - 1)
                            dd = min(max(z + k - cz, 0), depth - 1)
                            acc += image[rr, cc, dd] * kernel[i, j, k]
                frow[y, z] = acc
        shared_matrix[x, :, :] = frow
        return x  # return processed row index to report progress


    # Parallel processing
    def parallel_shared_image(self, num_process=4):
        r, c, d = self.shape
        shared_out = mp.Array('d', r * c * d, lock=False)

        with mp.Pool(
            processes=num_process,
            initializer=init_sharedarray,
            initargs=(shared_out, self.shape)
        ) as pool:
            # Use imap for incremental progress reporting
            for idx in pool.imap(self.edge_filter_row, range(r)):
                print(f"Finished row {idx+1}/{r}")

        return tonumpyarray(shared_out).reshape(self.shape)

