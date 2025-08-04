import streamlit as st
from src.model_utils import load_finetuned_model
from src.inference import predict_answer

st.set_page_config(page_title="QA System", layout="centered")
st.title("📘 Question Answering System")

context = st.text_area("📝 Enter context passage:")
question = st.text_input("❓ Enter your question:")

if st.button("Get Answer"):
    if context.strip() and question.strip():
        model, tokenizer = load_finetuned_model()
        answer = predict_answer(question, context, model, tokenizer)
        st.success(f"✅ Answer: {answer}")
    else:
        st.warning("Please provide both context and question.")
