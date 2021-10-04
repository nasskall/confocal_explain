import glob
import math
import os

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import backend as K
from PIL import Image
import cv2
from matplotlib import pyplot as plt


class GradCAM:
    # Adapted with some modification from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    def __init__(self, model, layerName=None):
        """
        model: pre-softmax layer (logit layer)
        """
        self.model = model
        self.layerName = layerName

        if self.layerName == None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")

    def compute_heatmap(self, image, classIdx, upsample_size, eps=1e-5):
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output]
        )
        # record operations for automatic differentiation

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOuts, preds) = gradModel(inputs)  # preds after softmax
            loss = preds[:, classIdx]

        # compute gradients with automatic differentiation
        grads = tape.gradient(loss, convOuts)
        # discard batch
        convOuts = convOuts[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))

        # compute weights
        weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)

        # Apply reLU
        cam = np.maximum(cam, 0)
        initial_cam = cam / np.max(cam)
        cam = cv2.resize(initial_cam, upsample_size, interpolation=cv2.INTER_LINEAR)
        # convert to 3D
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])

        return initial_cam, cam3


def overlay_gradCAM(img, cam3):
    cam3 = np.uint8(255 * cam3)
    cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
    new_img = 0.3 * cam3 + 0.5 * img
    return (new_img * 255.0 / new_img.max()).astype("uint8")


@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy

    return tf.nn.relu(x), grad


# Reference: https://github.com/eclique/keras-gradcam with adaption to tensorflow 2.0
class GuidedBackprop:
    def __init__(self, model, layerName=None):
        self.model = model
        self.layerName = layerName
        self.gbModel = self.build_guided_model()
        if self.layerName == None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply Guided Backpropagation")

    def build_guided_model(self):
        gbModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output]
        )
        layer_dict = [layer for layer in gbModel.layers[1:] if hasattr(layer, "activation")]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu

        return gbModel

    def guided_backprop(self, images, upsample_size):
        """Guided Backpropagation method for visualizing input saliency."""
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            outputs = self.gbModel(inputs)
        grads = tape.gradient(outputs, inputs)[0]
        saliency = cv2.resize(np.asarray(grads), upsample_size)
        return saliency


def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def show_gradCAMs(model, gradCAM, GuidedBP, img, decode={}):
    """
    model: softmax layer
    """
    upsample_size = (img.shape[1], img.shape[0])
    im = img_to_array(img)
    x = np.expand_dims(im, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    idx = preds.argmax()
    if len(decode) == 0:
        res = decode_predictions(preds)[0][0][1:]
    else:
        res = [decode[idx], preds]
    initial, cam3 = gradCAM.compute_heatmap(image=x, classIdx=idx, upsample_size=upsample_size)
    new_img = overlay_gradCAM(img, cam3)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    # Show guided GradCAM
    gb = GuidedBP.guided_backprop(x, upsample_size)
    guided_gradcam = deprocess_image(gb * cam3)
    guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)
    return initial, new_img, guided_gradcam, res


def calculate_decrease(img, patch, result):
    row = math.floor(patch / 20)
    col = patch % 20
    img_array = np.array(img)
    pixel_col1 = 16 * col
    pixel_col2 = 16 * (col + 1)
    pixel_row1 = 16 * row
    pixel_row2 = 16 * (row + 1)
    img_array[pixel_row1:pixel_row2, pixel_col1:pixel_col2] = np.random.normal(127, 127, 768).reshape(16, 16, 3)
    _, _, _, res = show_gradCAMs(model, retrained_gradCAM, retrained_guidedBP,
                                 img_array, decode={1: "Malignant", 0: "Benign"})
    if result[0] == "Malignant":
        final_res = 1
    else:
        final_res = 0
    new_res = [res[0] == result[0], res[1][0][final_res]]
    return img_array, new_res


total_results = []
for image_path in glob.glob('resources/*/*/*.bmp')[0:100]:
    image_s = Image.open(image_path)
    image_s = image_s.resize((320, 320))
    model = tf.keras.models.load_model('models/vgg16_model')
    model_logit = Model(model.input, model.layers[-2].output)
    retrained_gradCAM = GradCAM(model=model_logit, layerName="block5_conv3")
    retrained_guidedBP = GuidedBackprop(model=model, layerName="block5_conv3")
    initial, new_img, guidedcam_img, res = show_gradCAMs(model, retrained_gradCAM, retrained_guidedBP,
                                                         np.array(image_s), decode={1: "Malignant", 0: "Benign"})
    sorted_initial = np.argsort(initial, axis=None)
    sorted_list_rev = sorted_initial.tolist()[::-1]
    img_arr = image_s
    img_results = []
    for tile in sorted_list_rev:
        img_arr, new_resu = calculate_decrease(img_arr, tile, res)
        if new_resu[0] == image_path.split('\\')[1]:
            img_results.append(new_resu[1])
    img_arr_res = np.array(img_results)
    total_results.append(img_arr_res)
total_results_arr = np.array(total_results)
avg_results = np.average(total_results_arr, axis=0)
print(avg_results)
plt.plot(avg_results)
plt.show()
