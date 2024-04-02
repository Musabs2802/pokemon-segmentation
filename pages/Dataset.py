import streamlit as st
from pycaret.datasets import get_data
from pycaret.clustering import *
from PIL import Image


def run():
    image = Image.open("./res/hero_image.jpg")
    st.image(image, use_column_width=True)

    st.title("Pokemon Dataset")

    st.dataframe(get_data("pokemon"))


if __name__ == "__main__":
    run()
