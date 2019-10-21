import random, requests, math, sys, os
import numpy as np
from PIL import Image
from io import BytesIO
from scipy.ndimage.filters import gaussian_filter          
import tensorflow as tf
from inception import Inception
from utilities import save_image, resize_image, normalize_image

OUTPUT_IMAGE_NAME = "deep_dream"
ITERATIONS = 15
STEP_SIZE = 2.0
RESCALE_FACTOR = 0.7
LEVELS = 5
BLEND = 0.25
BLUR_SIGMA = 0.5
TILE_SIZE = 256


# Arbitrarily set layers to optimize
LAYER_INDICES = None # e.g [3, 5, 9]

# Or pick them randomly with the following bound parameters
MIN_OPERATIONS = 2
MAX_OPERATIONS = 5
MIN_LAYER = 2
MAX_LAYER = 10


class DeepDream:
    
    def __init__(self, model):
        self.model = model
        self.session = tf.InteractiveSession(graph=model.graph)

    def update_gradient(self, gradient, image):
        height, width = image.shape[:2]
        shifted_x, shifted_y = np.random.randint(TILE_SIZE, size=2)
        img_shift = np.roll(np.roll(image, shifted_x, 1), shifted_y, 0)
        updated_gradient = np.zeros_like(image)
        for y in range(0, max(height-TILE_SIZE//2, TILE_SIZE),TILE_SIZE):
            for x in range(0, max(width-TILE_SIZE//2, TILE_SIZE),TILE_SIZE):
                sub = img_shift[y:y+TILE_SIZE, x:x+TILE_SIZE]
                feed_dict = self.model.get_feed_dict(image=sub)
                new_gradient = self.session.run(gradient, feed_dict=feed_dict)
                new_gradient /= (np.std(new_gradient) + 1e-8)
                updated_gradient[y:y+TILE_SIZE,x:x+TILE_SIZE] = new_gradient
        return np.roll(np.roll(updated_gradient, -shifted_x, 1), -shifted_y, 0)
    
    def optimize_image(self, layer, image, iterations, step_size):
        image = image.copy()
        for i in range(iterations):
            gradient = self.update_gradient(self.model.get_gradient(layer), image)
            gradient = gaussian_filter(gradient, sigma=(BLUR_SIGMA, BLUR_SIGMA, 0))
            scaled_step_size = step_size / (np.std(gradient) + 1e-8)
            image += gradient * scaled_step_size
            print("iteration " + str(i) + " out of " + str(iterations), end="\r" if i != iterations-1 else "\r\033[K")
        return image
    
    def recursively_optimize(self, layer, image, levels, rescale_factor, blend, iterations, step_size):
        
        if levels > 0:
            blurred = gaussian_filter(image, sigma=(BLUR_SIGMA, BLUR_SIGMA, 0))

            downscaled = resize_image(image=blurred, factor=rescale_factor)

            final_image = self.recursively_optimize(layer=layer,
                                                    image=downscaled,
                                                    levels=levels-1,
                                                    rescale_factor=rescale_factor,
                                                    blend=blend,
                                                    iterations=iterations,
                                                    step_size=step_size)

            upscaled = resize_image(image=final_image, size=image.shape[0:2])
            
            image = blend * image + (1.0 - blend) * upscaled

        print("\rlevel " + str(levels) + " out of " + str(LEVELS))
        final_image = self.optimize_image(layer=layer,
                                          image=image,
                                          iterations=iterations,
                                          step_size=step_size)
        return final_image   
    
def main():
    if len(sys.argv) > 1:
        image_to_open = sys.argv[1]
    else:
        response = requests.get("https://picsum.photos/1080")
        image_to_open = BytesIO(response.content)
    image = np.float32(Image.open(image_to_open))

    model = Inception()
    
    if LAYER_INDICES is None:
        optimizations_to_perform = random.randrange(MIN_OPERATIONS, MAX_OPERATIONS)
        layer_indices = random.sample(range(MIN_LAYER, MAX_LAYER), optimizations_to_perform)
    else:
        layer_indices = LAYER_INDICES
        
    final_image = image
    deep_dream = DeepDream(model)
    for i, layer_index in enumerate(layer_indices):
        print("LAYER " + model.layer_names[layer_index] + ", " + str(i) + " out of " + str(len(layer_indices)))
        layer = model.layers[layer_index]
        final_image = deep_dream.recursively_optimize(layer=layer,
                                                      image=final_image,
                                                      iterations=ITERATIONS,
                                                      step_size=STEP_SIZE,
                                                      rescale_factor=RESCALE_FACTOR,
                                                      levels=LEVELS,
                                                      blend=BLEND)
    save_image(final_image, OUTPUT_IMAGE_NAME + ".jpeg")
  
if __name__== "__main__":
    main()
 