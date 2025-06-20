import streamlit as st
import pandas as pd
from regex_class import classify_with_regex
from bert_class import classify_with_bert
from llm_classify import classify_with_llm

# Core logic
def classify(logs):
    labels = []
    for source, log_msg in logs:
        label = classify_log(source, log_msg)
        labels.append(label)
    return labels

def classify_log(source, log_msg):
    if source == "LegacyCRM":
        label = classify_with_llm(log_msg)
    else:
        label = classify_with_regex(log_msg)
        if not label:
            label = classify_with_bert(log_msg)
    return label

# Streamlit UI
st.set_page_config(page_title="Log Message Classifier", layout="wide")

st.title("📄 Log Message Classifier")
st.write("Upload a CSV file with `source` and `log_message` columns to see classification results.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if "source" not in df.columns or "log_message" not in df.columns:
            st.error("CSV must contain 'source' and 'log_message' columns.")
        else:
            logs = list(zip(df["source"], df["log_message"]))
            labels = classify(logs)
            df["Predicted Label"] = labels

            st.success("✅ Classification completed!")
            st.dataframe(df[["source", "log_message", "Predicted Label"]], use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
