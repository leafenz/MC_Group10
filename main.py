from myfunction import Filter_three_Dimension, init_sharedarray, tonumpyarray
import numpy as np
import multiprocessing as mp

if __name__ == "__main__":
    mp.freeze_support()  # Mandatory in Windows!

    image = np.random.rand(10, 10, 3) # Creating an image in this case to try it out
    kernel = np.ones((3, 3, 3)) / 27

    f3d = Filter_three_Dimension(image, kernel)

    print("🔧 Applying parallel 3D filter...")
    result = f3d.parallel_shared_image(num_process=4)

    print("✅ Filtering completed.")
    print("Forma de salida:", result.shape)
    print("Primeros valores de la primera fila:\n", result[0, :, 0])

