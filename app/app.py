import streamlit as st
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

MODEL_PATH = "vit-dog-breed"

st.set_page_config(page_title="Dog Breed Identifier", page_icon="🐶")


@st.cache_resource
def load_model():
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
    model = ViTForImageClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return processor, model


def predict(image: Image.Image, processor, model, top_k=3):
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    top_probs, top_idxs = torch.topk(probs, top_k)

    results = []
    for prob, idx in zip(top_probs, top_idxs):
        label_key = idx.item()
        if label_key not in model.config.id2label:
            label_key = str(label_key)
        breed = model.config.id2label[label_key]
        results.append((breed, prob.item()))
    return results


def main():
    st.title("🐶 Dog Breed Identifier")
    st.write(
        "Upload a photo of a dog and this ViT model (fine-tuned on the "
        "Stanford Dogs dataset) will guess the breed."
    )

    processor, model = load_model()

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with col2:
            with st.spinner("Predicting..."):
                results = predict(image, processor, model)

            st.subheader("Top predictions")
            for breed, prob in results:
                st.write(f"**{breed}**")
                st.progress(prob)
                st.caption(f"{prob * 100:.1f}% confidence")

    st.divider()
    st.caption(
        "Model: google/vit-base-patch16-224, fine-tuned on Stanford Dogs "
        "(120 breeds). Built for a Deep Learning for Images course project."
    )


if __name__ == "__main__":
    main()
