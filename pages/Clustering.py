import streamlit as st
from pycaret.datasets import get_data
from pycaret.clustering import *
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

model = load_model("../../output/clustering_model")

def plot_features(data, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(f"Distribution of {title}")
    st.pyplot(fig)
    
def run():
    image = Image.open("../../res/hero_image.jpg")
    st.image(image, use_column_width=True)

    st.title("Pokemon Clustering Analysis")

    df = get_data("pokemon")
    df.drop(["#", "Name"], axis=1, inplace=True)
    setup(data=df)
    kmeans = create_model("kmeans", 4)
    kmeans_result = assign_model(kmeans)

    st.header("Cluster plot")
    plot_model(kmeans, display_format="streamlit")

    st.header("Distribution plot")
    plot_model(kmeans, plot="distribution", display_format="streamlit")

    st.header("Properties of Cluster 0")
    plot_features(kmeans_result[kmeans_result["Cluster"] == "Cluster 0"], "Cluster 0")

    st.header("Properties of Cluster 1")
    plot_features(kmeans_result[kmeans_result["Cluster"] == "Cluster 1"], "Cluster 1")

    st.header("Properties of Cluster 2")
    plot_features(kmeans_result[kmeans_result["Cluster"] == "Cluster 2"], "Cluster 2")

    st.header("Properties of Cluster 3")
    plot_features(kmeans_result[kmeans_result["Cluster"] == "Cluster 3"], "Cluster 3")

if __name__ == "__main__":
    run()
