import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import Algorithmia

client = Algorithmia.client(st.secrets["algorithmia_access"])
algo = client.algo('shadyvale/birdsong_classifier/0.1.1')

from algo import apply

st.title("UK birdsong classifier demo")

st.text("upload a file of a birdsong recording to get the top 5 most likely birds")
uploaded_file = st.file_uploader("Choose a file of your recording (only wav supported at the moment)")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    wav, sample_rate = tf.audio.decode_wav(
        bytes_data,
        desired_channels=1)
    st.write(f"sample rate of audio: {sample_rate}")
    response = apply(bytes_data)
    st.write(response)