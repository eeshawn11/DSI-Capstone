# import libraries
import streamlit as st
import os

st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Report a bug": "https://github.com/eeshawn11/DSI-Capstone/issues",
        "About": "Thanks for dropping by!"
        }
    )

if "WORKSPACE_PATH" not in st.session_state:
    st.session_state.WORKSPACE_PATH = os.getcwd()

with st.sidebar:
    st.markdown(
        """
        Created by [**eeshawn**](https://eeshawn.com)

        - Connect on [**LinkedIn**](https://www.linkedin.com/in/shawn-sing/)
        - Project source [**code**](https://github.com/eeshawn11/DSI-Capstone/)
        - Check out my other projects on [**GitHub**](https://github.com/eeshawn11/)
        """
        )

st.markdown(
    """
    <div style='text-align:center'>
    <img src='https://live.staticflickr.com/65535/51052311322_d9236f488f_c.jpg' width=799px height=533px>
    <p>Photo by <a href='https://www.flickr.com/photos/ee_shawn/' target='_blank'>Ee Shawn</a> on <a href='https://flickr.com/photos/ee_shawn/51052311322/' target='_blank'>Flickr</a></p>
    </div>
    """, unsafe_allow_html=True
    )
st.markdown(
    """
    ## Introduction

    Image caption generation allows a machine to generate a sentence in natural language that describes the context of an image. While seemingly mundane, this could have wider applications in various fields, such as improving accessibility for the visually impaired by providing descriptions of images, or even in healthcare by automatically generating captions for X-ray images to aid doctors in diagnosis.

    Personally, as a photography hobbyist, when sharing my [photographs](https://flickr.com/photos/ee_shawn/), I like to pair them with some text that perhaps provides some context to the image and enhances appreciation. It would be interesting if I could train a model that analyses an image and automatically generates a descriptive caption providing some history or information about the image based on the location. This capstone project serves as a foundation: generating a simple descriptive caption based on features within the image.
    """
)

st.markdown(
    """
    ### Problem Statement

    To train an attention mechanism-based caption generator that is able to generate a descriptive caption of an image with a BLEU-1 score of at least 0.5.
    """
)

st.markdown(
    """
    ### Model

    The model architecture used here is based on the TensorFlow [tutorial](https://www.tensorflow.org/tutorials/text/image_captioning), inspired by [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044) and updated to use a 2-layer Transformer-decoder.

    The model has been trained on the Flickr30k [dataset](https://www.kaggle.com/datasets/eeshawn/flickr30k), achieving a BLEU-1 score of 0.52. The model uses the MobileNetV3-Small as a feature extractor and a Tokenizer with a vocabulary of 6,000 words.

    ### Background

    It has been a fantastic learning journey over the course of a 12-week Data Science Immersive with General Assembly, from learning the basics of Python to getting acquainted with machine learning through scikit-learn. With the prescribed course projects covering various aspects of machine learning such as regression and classification problems, I wanted to venture further and explore deep learning with larger datasets. With a variety of tutorials available, TensorFlow and Keras seemed like a good place to start.

    ### Limitations

    As with all projects, there are limitations to the current deployed model. While relatively robust for simpler use cases like my project, the Flickr30k dataset only includes around 31,000 image samples. The model would be constrained by the types of images and captions present in the dataset, as you may observe with the predictions.
        
    To fully maximise the potential of the model, larger datasets like COCO or Conceptual Captions could be considered.
    """
)
