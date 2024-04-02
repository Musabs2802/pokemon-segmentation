import streamlit as st
import pandas as pd
from pycaret.classification import *
from PIL import Image

model = load_model("../../output/classification_model")

st.cache_data.clear()


def predict(data):
    predict_result = predict_model(model, data=data)
    return predict_result["prediction_label"][0]


def run():
    image = Image.open("../../res/hero_image.jpg")
    st.image(image, use_column_width=True)

    select_box = st.sidebar.selectbox("Choose predict method:", ("Single", "Bulk"))
    st.sidebar.info("This app segments pokemon based on their stats")

    st.title("Pokemon Classification App")

    if select_box == "Single":
        name = st.text_input("Name")

        col1, col2 = st.columns(2)
        type1 = col1.selectbox(
            "Type 1",
            [
                "Grass",
                "Fire",
                "Water",
                "Bug",
                "Normal",
                "Poison",
                "Electric",
                "Ground",
                "Fairy",
                "Fighting",
                "Psychic",
                "Rock",
                "Ghost",
                "Ice",
                "Dragon",
                "Dark",
                "Steel",
                "Flying",
            ],
        )
        type2 = col2.selectbox(
            "Type 2",
            [
                "Poison",
                None,
                "Flying",
                "Dragon",
                "Ground",
                "Fairy",
                "Grass",
                "Fighting",
                "Psychic",
                "Steel",
                "Ice",
                "Rock",
                "Dark",
                "Water",
                "Electric",
                "Fire",
                "Ghost",
                "Bug",
                "Normal",
            ],
        )
        total = st.slider("Total", min_value=100, max_value=1000)

        col1, col2 = st.columns(2)
        hp = col1.slider("HP", min_value=1, max_value=500)
        attack = col2.slider("Attack", min_value=1, max_value=500)

        col1, col2 = st.columns(2)
        defense = col1.slider("Defense", min_value=1, max_value=500)
        sp_attk = col2.slider("Sp. Attack", min_value=1, max_value=500)

        col1, col2 = st.columns(2)
        sp_defense = col1.slider("Sp. Defense", min_value=1, max_value=500)
        speed = col2.slider("Speed", min_value=1, max_value=500)

        col1, col2 = st.columns(2)
        generation = col1.number_input("Generation", min_value=1, max_value=6)
        legendary = col2.radio("Legendary", [True, False])

        data = pd.DataFrame(
            [
                {
                    "Type 1": type1,
                    "Type 2": type2,
                    "Total": total,
                    "HP": hp,
                    "Attack": attack,
                    "Defense": defense,
                    "Sp. Atk": sp_attk,
                    "Sp. Def": sp_defense,
                    "Speed": speed,
                    "Generation": generation,
                    "Legendary": legendary,
                }
            ]
        )

        output = ""
        if st.button("Predict", type="primary"):
            output = predict(data)

        if output:
            st.success("This belong's to {}".format(output))
            st.page_link(
                label="See how {} performs".format(output),
                page="./pages/Clustering.py",
            )

    if select_box == "Bulk":
        file_upload = st.file_uploader("Upload csv file", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(model, data=data)
            st.write(predictions)


if __name__ == "__main__":
    run()
