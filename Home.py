# import libraries
import streamlit as st
import os

import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

if "WORKSPACE_PATH" not in st.session_state:
    st.session_state.WORKSPACE_PATH = os.getcwd()

with st.sidebar:
    st.markdown(
        """
        Created by Shawn

        - Happy to connect on [LinkedIn](https://www.linkedin.com/in/shawn-sing/)
        - Check out my other projects on [GitHub](https://github.com/eeshawn11/)
        """
        )

st.markdown("""
    ## Introduction

    It has been a fantastic learning journey over the course of the 12-week Data Science Immersive with General Assembly, from learning the basics of Python to getting acquainted with machine learning through scikit-learn. With the prescribed course projects covering various aspects of machine learning such as regression and classification problems, I wanted to venture further and explore deep learning with larger datasets. With a variety of tutorials available, TensorFlow and Keras seemed like a good place to start.

    Image caption generation allows a machine to generate a sentence in natural language that describes the context of an image. While seemingly mundane, this could have wider applications in various fields, such as assisting the visually impaired by explaining images through text-to-speech systems or perhaps used to tag and organise photos in a library.

    Personally, as a photography hobbyist, when sharing my [photographs](https://flickr.com/photos/ee_shawn/), I like to pair them with some text that perhaps provides some context to the image and enhances appreciation. It would be interesting if I could train a model that analyses an image and automatically generates a descriptive caption providing some history or information about the image based on the location.

    This would likely be a rather complex task, so to start off with the first step into my deep learning journey: training a supervised model that is capable of generating a simple caption based on detected objects within the image.

    ### Background

    Attention is a cognitive ability that humans have to selectively concentrate on a discrete aspect of information while ignoring other perceivable information. When applied to deep learning, the use of an attention mechanism similarly allows a machine to focus on the most relevant parts of a sequence when making a prediction.

    The model architecture used here is based on the TensorFlow [tutorial](https://www.tensorflow.org/tutorials/text/image_captioning), inspired by [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044) and updated to use a 2-layer Transformer-decoder.

    """)