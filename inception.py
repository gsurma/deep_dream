import numpy as np
import tensorflow as tf

class Inception:

    input_image = "input:0"
    layer_names = ["conv2d0", "conv2d1", "conv2d2",
                   "mixed3a", "mixed3b",
                   "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e",
                   "mixed5a", "mixed5b"]

    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.gfile.FastGFile("inception/5h/tensorflow_inception_graph.pb", "rb") as file:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(file.read())
                tf.import_graph_def(graph_def, name="")
            self.input = self.graph.get_tensor_by_name(self.input_image)
            self.layers = [self.graph.get_tensor_by_name(name + ":0") for name in self.layer_names]

    def get_feed_dict(self, image=None):
        image = np.expand_dims(image, axis=0)
        feed_dict = {self.input_image: image}
        return feed_dict

    def get_gradient(self, tensor):
        with self.graph.as_default():
            tensor = tf.square(tensor)
            tensor_mean = tf.reduce_mean(tensor)
            gradient = tf.gradients(tensor_mean, self.input)[0]
        return gradient
