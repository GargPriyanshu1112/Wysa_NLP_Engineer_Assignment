import json
import streamlit as st
from transformers import pipeline


def get_corresponding_cls(cls_to_lbl_map, pred_lbl):
    lbl_to_cls_map = {lbl: cls for cls, lbl in cls_to_lbl_map.items()}
    return lbl_to_cls_map[pred_lbl]


# Streamlit app
def main(mappings):
    # Page title
    st.title("Emotion Analyzer")
    # User input
    input_text = st.text_input("Enter your text here:")

    if st.button("Submit"):
        if input_text:
            with st.spinner(
                "Loading models... This will take a moment, and only happens once."
            ):
                emotion_recog_model = pipeline(
                    "text-classification", model="PriyHF/emotion_recog"
                )
                brand_product_recog_model = pipeline(
                    "text-classification", model="PriyHF/brand_product_recog"
                )

                emotion_pred_dict = emotion_recog_model(input_text)[0]
                emotion_pred_lbl = emotion_pred_dict["label"]
                emotion_pred_lbl = int(emotion_pred_lbl.split("_")[-1])
                emotion_pred_cls = get_corresponding_cls(
                    mappings["emotions"], emotion_pred_lbl
                )

                st.subheader("Predicted Emotion: ")
                st.write(emotion_pred_cls)

                if emotion_pred_cls in ["Positive emotion", "Negative emotion"]:
                    brand_prod_pred_dict = brand_product_recog_model(input_text)[0]
                    brand_prod_pred_lbl = brand_prod_pred_dict["label"]
                    brand_prod_pred_lbl = int(brand_prod_pred_lbl.split("_")[-1])
                    brand_product_pred_cls = get_corresponding_cls(
                        mappings["categories"], brand_prod_pred_lbl
                    )
                    st.subheader("Category: ")
                    st.write(brand_product_pred_cls)
        else:
            st.error("Please enter some text!")


if __name__ == "__main__":
    # Get mappings
    with open(r"D:\Priyanshu\Wysa\code\mappings\mapping.json", "r") as file:
        mappings = json.load(file)
    file.close()
    # Initiate
    main(mappings)
