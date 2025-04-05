import gradio as gr
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Prediction function
def predict_spam(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "❌ Spam" if pred == 1 else "✅ Not Spam"

# Gradio interface
iface = gr.Interface(fn=predict_spam, inputs="text", outputs="text", title="Spam Detector")

# Launch if running locally
if __name__ == "__main__":
    iface.launch()
