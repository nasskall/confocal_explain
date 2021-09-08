# This is a sample Python script.
import cv2
from tensorflow.keras import backend as K
import os
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.vgg16 import preprocess_input
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from utils import SessionState  # Assuming SessionState.py lives on this folder

session = SessionState.get(run_id=0)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('models/vgg16_model')
    return model

def main():
    model = load_model()
    image_s = None
    st.title("Explaining predictions on skin confocal images")
    seg_algo = st.sidebar.radio('Type of segmentation algorithm', ('Felzenswalb', 'Slic', 'Quickshift'))
    with st.container():
                st.title("VGG16 and " + seg_algo)
                st.sidebar.write('You selected VGG16 and ' + seg_algo)
                # Add a slider to the sidebar:
                imp_threshold,params = get_params(seg_algo)
                sample = st.file_uploader("Choose an sample image...")
                if sample is not None:
                    image_s = Image.open(sample)
                    image_s = image_s.resize((500,500))
                    st.image(image_s, caption='Sample Image', width=300)
                if sample is not None:
                    if st.button('Predict and Explain'):
                        with st.spinner("Explaining predictive model's results"):
                            image_s = np.array(image_s)
                            model_logit = Model(model.input, model.layers[-2].output)
                            retrained_gradCAM = GradCAM(model=model_logit, layerName="block5_conv3")
                            retrained_guidedBP = GuidedBackprop(model=model, layerName="block5_conv3")
                            cam, new_img, guidedcam_img, res = show_gradCAMs(model, retrained_gradCAM,
                                                                             retrained_guidedBP,
                                                                             image_s,
                                                                             decode={1: "Malignant", 0: "Benign"})
                            output_image1 = Image.fromarray(new_img)
                            output_image2 = Image.fromarray(guidedcam_img)
                            dst, image_c = generate_maps(image_s, cam, imp_threshold, params, seg_algo)
                            if output_image1:
                                st.header("Prediction: {}".format(res[0]))
                                st.header("Probability: {:.2f} for VGG19".format(float(res[1])))
                                col_gc, col_gcc = st.columns(2)
                                with col_gc:
                                    st.subheader("Grad-CAM")
                                    st.image(output_image1)
                                with col_gcc:
                                    st.subheader("Guided Grad-CAM")
                                    st.image(output_image2)
                                col_gc1, col_gcc1 = st.columns(2)
                                with col_gc1:
                                    st.subheader("Important Regions")
                                    st.image(image_c)
                                with col_gcc1:
                                    st.subheader("Grad CAM and " + seg_algo)
                                    st.image(dst)
                                st.success('Done')
                        if st.button(
                                'Try again'):
                            session.run_id += 1

def get_params(seg_algo=None):
    imp_threshold = st.sidebar.slider(
        'Importance threshold',
        0.01, 1.0, 0.3)
    if seg_algo == 'Felzenswalb':
        scale = st.sidebar.slider(
            'Scale',
            0, 1000, 100)
        sigma = st.sidebar.slider(
            'Sigma',
            0.01, 2.00, 0.5)
        min_size = st.sidebar.slider(
            'Minimum size',
            0, 2000, 50)
        params = [scale, sigma, min_size]
    elif seg_algo == 'Slic':
        n_segments = st.sidebar.slider(
            'Number of segments',
            1, 1000, 250)
        compactness = st.sidebar.slider(
            'Compactness',
            0.1, 100.0, 10.0)
        sigma = st.sidebar.slider(
            'Sigma',
            0.1, 10.0, 1.0)
        params = [n_segments, compactness, sigma]
    elif seg_algo == 'Quickshift':
        kernel_size = st.sidebar.slider(
            'Kernel size',
            1, 10, 3)
        max_dist = st.sidebar.slider(
            'Maximum distance',
            1, 20, 6)
        ratio = st.sidebar.slider(
            'Ratio',
            0.1, 1.0, 0.5)
        params = [kernel_size, max_dist, ratio]
    return imp_threshold,params


def generate_maps(image_s, class_act, imp_thre, params,seg_algo=None):
    if seg_algo == 'Felzenswalb':
        segments = felzenszwalb(image_s, scale=params[0], sigma=params[1], min_size=params[2])
        st.header(f"Felzenszwalb number of segments: {len(np.unique(segments))}")
    elif seg_algo == 'Slic':
        segments = slic(image_s, n_segments=params[0], compactness=params[1], sigma=params[2],
                        start_label=1, slic_zero=True)
        st.header(f"SLIC number of segments: {len(np.unique(segments))}")
    elif seg_algo == 'Quickshift':
        segments = quickshift(image_s, kernel_size=params[0], max_dist=params[1], ratio=params[2])
        st.header(f"Quickshift number of segments: {len(np.unique(segments))}")

    image_hsv = cv2.cvtColor(class_act, cv2.COLOR_RGB2HSV)
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 10])
    upper1 = np.array([2, 255, 255])
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([168, 100, 10])
    upper2 = np.array([180, 255, 255])
    lower_mask = cv2.inRange(image_hsv, lower1, upper1)
    upper_mask = cv2.inRange(image_hsv, lower2, upper2)
    full_mask = lower_mask + upper_mask;
    hsv_threshold = cv2.bitwise_not(full_mask)
    contours, hierarchy = cv2.findContours(hsv_threshold, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    k = -1
    image_c = image_s.copy()
    for i, cnt in enumerate(contours):
        if (hierarchy[0, i, 3] == -1):
            k += 1
        cv2.drawContours(image_c, [cnt], -1, (255, 0, 0), 1)
    img = image_s
    for i in range(len(np.unique(segments))):
        seg_pixels = np.where(segments == i)
        seg_list = list(zip(seg_pixels[0], seg_pixels[1]))
        count = 0
        intensity = np.zeros(3)
        for pixel in seg_list:
            if hsv_threshold[pixel[0], pixel[1]] == 255:
                int_value = class_act[pixel[0], pixel[1]]
                intensity += int_value
                count += 1
        if count != 0:
            print((intensity * (1 / count)).astype(int))
            image_r = mark_boundaries(img, (segments == i).astype(int),
                                      background_label=0,
                                      color=(intensity * (1 / count)).astype(int),
                                      mode='inner')
            if count > imp_thre * len(seg_list):
                img = image_r
    img_mask = img / 255
    dst = cv2.addWeighted(image_s / 255, 0.6, img_mask, 0.4, 0)
    return dst, image_c


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
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, upsample_size, interpolation=cv2.INTER_LINEAR)

        # convert to 3D
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])

        return cam3


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
        res = [decode[idx], preds.max()]
    cam3 = gradCAM.compute_heatmap(image=x, classIdx=idx, upsample_size=upsample_size)
    cam, new_img = overlay_gradCAM(img, cam3)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    # Show guided GradCAM
    gb = GuidedBP.guided_backprop(x, upsample_size)
    guided_gradcam = deprocess_image(gb * cam3)
    guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)
    return cam, new_img, guided_gradcam, res


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
