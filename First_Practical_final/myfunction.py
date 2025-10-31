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
    return np.frombuffer(mp_arr.get_obj(), dtype=np.float32)


# ===============================================================
# Initializer (executed once per worker)
# ===============================================================
def init_sharedarray(shared_array, img_shape, image=None, kernel=None):
    global shared_space
    global shared_matrix
    
    
    shared_space = shared_array 
    
    # ensure shape is alway 3D for shared_matrix
    if len(img_shape) == 2:
        img_shape = (img_shape[0], img_shape[1], 1)
    elif len(img_shape) >3:
        # higher - dimensional -> squeeze extras into last axis
        new_depth = int(np.prod(img_shape[2:]))
        img_shape = (img_shape[0], img_shape[1], new_depth)

    shared_matrix = tonumpyarray(shared_space).reshape(img_shape)


# ===============================================================
# Filter Class
# ===============================================================
class Filter_three_Dimension:
    def __init__(self, image, imagefilter):
        self.image = image.astype(np.float32)
        self.imagefilter = imagefilter.astype(np.float32)
        self.shape = image.shape

    # Worker function (runs in parallel)
    def edge_filter_row(self, x):
        global shared_matrix
        image = self.image
        kernel = self.imagefilter
        (rows, cols, depth) = image.shape
        (kx, ky, kz) = kernel.shape
        cx, cy, cz = kx // 2, ky // 2, kz // 2

        # we use this temporary local buffer for this row
        frow = np.zeros((cols, depth), dtype=np.float32)

        # convolution for one x- slice
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

        # safe write to shared memory with lock
        with shared_space.get_lock():
            shared_matrix[x, :, :] = frow
        return x  # return processed row index to report progress







  

