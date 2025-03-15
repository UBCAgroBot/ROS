import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

def preprocess_image(self, image_path):
    """
    Preprocess the image: resize, normalize, and convert to the required format for the model.
    :param image_path: Path to the image file.
    :return: Preprocessed image ready for inference, original image, and scaling factors.
    """
    # Decode & resize image
    img = cv2.imread(image_path)
    original_size = img.shape[:2]  # Original size (height, width)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, self.input_size)

    # Conversion to float32 for CUDA compatibility
    resized = resized.astype(np.float32)

    # Allocate memory on GPU for I/O
    img_gpu = drv.mem_alloc(resized.nbytes)
    drv.memcpy_htod(img_gpu, resized)
    normalized_gpu = drv.mem_alloc(resized.nbytes)

    # CUDA Normalization
    mod = SourceModule("""
    __global__ void normalize(float *img, float *out, int width, int height, int channels){
        // (x,y) coordinate of thread
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (x < width && y < height){
            int i = (y * width + x) * channels; // Row major
                       
            for (int channel=0; channel<channels; channel++){
                float pixel = img[i+channel];
                out[i+channel] = ((pixel/255.0) - 0.5)/0.5
            }
        }
    }
    """)
    normalize = mod.get_function("normalize")

    # Set block/grid dimensions
    width, height = self.input_size
    channels = 3 # (R,G,B)
    block = (16,16,1) #(x,y,z)
    blocks_x = (width + block[0]-1) // block[0]     # Ceiling division
    blocks_y = (height + block[1]-1) // block[1]    # Ceiling division
    grid = (blocks_x, blocks_y)

    # Launch kernel
    normalize(img_gpu, normalized_gpu, width, height, channels, block=block, grid=grid)

    # Move normalized image back to CPU memory
    normalized = np.empty_like(resized)
    drv.memcpy_htod(normalized, normalized_gpu)

    # HWC to CHW format for model input
    input_tensor = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension

    # Compute scaling factors to map back to original size
    scale_x = original_size[1] / self.input_size[0]
    scale_y = original_size[0] / self.input_size[1]

    return input_tensor, img, (scale_x, scale_y)

'''
def preprocess_image(self, image_path):
    """
    Preprocess the image: resize, normalize, and convert to the required format for the model.
    :param image_path: Path to the image file.
    :return: Preprocessed image ready for inference, original image, and scaling factors.
    """
    img = cv2.imread(image_path)
    original_size = img.shape[:2]  # Original size (height, width)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, self.input_size)
    # Normalize the image (assuming mean=0.5, std=0.5 for demonstration)
    normalized = resized / 255.0
    normalized = (normalized - 0.5) / 0.5
    # HWC to CHW format for model input
    input_tensor = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension

    # Compute scaling factors to map back to original size
    scale_x = original_size[1] / self.input_size[0]
    scale_y = original_size[0] / self.input_size[1]

    return input_tensor, img, (scale_x, scale_y)
'''