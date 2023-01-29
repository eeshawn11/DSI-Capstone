# import libraries
import os
import re
import string
import streamlit as st
from pickle import load
from PIL import Image

import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from custom_model import Captioner, TokenOutput

max_caption_length = 73
MODEL_PATH = os.path.join(st.session_state.WORKSPACE_PATH, "models")
VOCAB_PATH = os.path.join(st.session_state.WORKSPACE_PATH, "tokenizer_vocab.pkl")
OUTPUT_LAYER_PATH = os.path.join(st.session_state.WORKSPACE_PATH, "output_layer_bias.pkl")

def load_image(image_path, image_shape=(224, 224, 3), preserve_aspect_ratio=False):
    if type(image_path) == str:
        img = tf.io.read_file(image_path)
        img = tf.io.decode_image(img, channels=3)
    else:
        img = image_path
    img = tf.image.resize(img, image_shape[:-1], preserve_aspect_ratio=preserve_aspect_ratio)
    return img

@st.experimental_singleton(show_spinner="Model building in progress...", max_entries=1)
def build_model():
    # image encoder
    feature_extractor = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        include_preprocessing=True,
        input_shape=(224, 224, 3),
    )

    # tokenizer
    def standardize(s):
        s = tf.strings.lower(s)
        s = tf.strings.regex_replace(s, f"[{re.escape(string.punctuation)}]", "")
        s = tf.strings.join(["[START]", s, "[END]"], separator=" ")
        return s

    with open(VOCAB_PATH, 'rb') as f:
        vocab = load(f)
    # load original vocab
    tokenizer = tf.keras.layers.TextVectorization(
        vocabulary=vocab, standardize=standardize, ragged=True
    )

    with open(OUTPUT_LAYER_PATH, 'rb') as f:
        output_layer_bias = load(f)

    output_layer = TokenOutput(tokenizer, banned_tokens=("", "[UNK]", "[START]"))
    output_layer.set_bias(output_layer_bias)

    model = Captioner(
        tokenizer,
        feature_extractor=feature_extractor,
        output_layer=output_layer,
        units=256,
        max_length=max_caption_length,
        dropout_rate=0.5,
        num_layers=2,
        num_heads=2,
    )

    checkpoint = tf.train.Checkpoint(model)
    checkpoint.restore(MODEL_PATH + "/attention_weights/").expect_partial()

    return model

model = build_model()

with st.sidebar:
    st.markdown(
        """
        Created by Shawn

        - Happy to connect on [LinkedIn](https://www.linkedin.com/in/shawn-sing/)
        - Check out my other projects on [GitHub](https://github.com/eeshawn11/)
        """
        )

###
# Include sample images for selection
# Include option to generate by URL?
###

results_placeholder = st.empty()

uploaded_file = st.file_uploader(
    "Upload image an image to generate a caption",
    type=["PNG", "JPG", "JPEG"],
)

if uploaded_file:
    with results_placeholder.container():
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        uploaded_image = load_image(uploaded_image)
        pred = model.simple_gen(uploaded_image)

        st.markdown("### Predicted Caption")
        st.success(pred)
        st.image(uploaded_file)
        st.markdown("---")
