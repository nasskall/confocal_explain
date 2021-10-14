import glob
import math
import os
import random

import pandas as pd
import tensorflow as tf
import numpy as np
from skimage.segmentation import felzenszwalb, mark_boundaries, slic, quickshift
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
    return cam3, (new_img * 255.0 / new_img.max()).astype("uint8")


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
    class_act, new_img = overlay_gradCAM(img, cam3)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    # Show guided GradCAM
    gb = GuidedBP.guided_backprop(x, upsample_size)
    guided_gradcam = deprocess_image(gb * cam3)
    guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)
    return class_act, initial, new_img, guided_gradcam, res


def calculate_decrease(img, segments, seg_place, result):
    inter_pixels = np.where(segments == seg_place)
    inter_pixel_list = list(zip(inter_pixels[0], inter_pixels[1]))
    img_array = np.array(img)
    for pix in inter_pixel_list:
        img_array[pix[0], pix[1]] = np.random.normal(127, 127, 3)
    im = img_to_array(img)
    x = np.expand_dims(im, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    idx = preds.argmax()
    decode = {1: "Malignant", 0: "Benign"}
    if len(decode) == 0:
        res = decode_predictions(preds)[0][0][1:]
    else:
        res = [decode[idx], preds]
    if result[0] == "Malignant":
        final_res = 1
    else:
        final_res = 0
    new_res = [res[0] == result[0], res[1][0][final_res]]
    return img_array, new_res, final_res


total_results_morf = []
total_results_aopc = []
image_path_list = glob.glob('resources/*/*/*.bmp')
random.shuffle(image_path_list)
for image_path in image_path_list:
    print(image_path)
    image_s = Image.open(image_path)
    image_s = image_s.resize((320, 320))
    model = tf.keras.models.load_model('models/vgg16_model')
    model_logit = Model(model.input, model.layers[-2].output)
    retrained_gradCAM = GradCAM(model=model_logit, layerName="block5_conv3")
    retrained_guidedBP = GuidedBackprop(model=model, layerName="block5_conv3")
    cls_act, initial, new_img, guidedcam_img, res = show_gradCAMs(model, retrained_gradCAM, retrained_guidedBP,
                                                                  np.array(image_s),
                                                                  decode={1: "Malignant", 0: "Benign"})
    #segments = slic(image_s, n_segments=250, compactness=10, sigma=1,
    #                    start_label=1, slic_zero=True)
    #segments = felzenszwalb(image_s, scale=100,sigma=0.5,min_size=50)
    segments = quickshift(image_s, kernel_size=3, max_dist=6, ratio=0.5)
    imp_thre = 0.55
    image_hsv = cv2.cvtColor(cls_act, cv2.COLOR_RGB2HSV)
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 10])
    upper1 = np.array([10, 255, 255])
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160, 100, 10])
    upper2 = np.array([180, 255, 255])
    lower_mask = cv2.inRange(image_hsv, lower1, upper1)
    upper_mask = cv2.inRange(image_hsv, lower2, upper2)
    full_mask = lower_mask + upper_mask
    hsv_threshold = full_mask
    img = image_s
    intensities = []
    for i in range(len(np.unique(segments))):
        seg_pixels = np.where(segments == i)
        seg_list = list(zip(seg_pixels[0], seg_pixels[1]))
        count = 0
        intensity = 0
        for pixel in seg_list:
            if hsv_threshold[pixel[0], pixel[1]] == 255:
                int_value = initial[math.floor(pixel[0] / 20), pixel[1] % 20]
                intensity += int_value
                count += 1
        if count > imp_thre * len(seg_list):
            intensities.append(intensity * (1 / count))
        else:
            intensities.append(0)
    sorted_int = np.argsort(intensities, axis=None)
    sorted_list_rev = sorted_int.tolist()[::-1]
    img_arr = image_s
    img_results_morf = []
    img_results_aopc = []
    if res[0]:
        for seg in sorted_list_rev:
            img_arr, new_resu, ind_c = calculate_decrease(img_arr, segments, seg, res)
            img_results_morf.append(new_resu[1])
            img_results_aopc.append(res[1][0][ind_c]-new_resu[1])
    img_arr_res_morf = np.array(img_results_morf)
    img_arr_res_aopc = np.array(img_results_aopc)
    img_arr_padded_morf = np.pad(img_arr_res_morf, (0, (700 - len(img_arr_res_morf))), 'constant', constant_values=(0, 0))
    img_arr_padded_aopc = np.pad(img_arr_res_aopc, (0, (700 - len(img_arr_res_morf))), 'constant', constant_values=(0, 0))
    total_results_morf.append(img_arr_padded_morf)
    total_results_aopc.append(img_arr_padded_aopc)
total_results_arr_morf = np.array(total_results_morf)
total_results_arr_aopc = np.array(total_results_aopc)
avg_results_morf = np.average(total_results_arr_morf, axis=0)
morf = np.average(avg_results_morf)
print(avg_results_morf)
avg_results_aopc = np.average(total_results_arr_aopc, axis=0)
aopc = np.sum(avg_results_aopc)*1/(401)
print(avg_results_aopc)
plt.plot(avg_results_morf[0:400])
plt.plot(avg_results_aopc[0:400])
plt.show()
df_res = pd.DataFrame([avg_results_morf, avg_results_aopc])
df_res.to_csv('df_res_quick_03_700.csv')