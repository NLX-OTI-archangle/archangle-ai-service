import gradio as gr
import joblib
import pandas as pd

model = joblib.load("models/isolation_forest.pkl")

def predict(step, amount, newBalInitiator, oldBalRecipient):
    X_new = pd.DataFrame(
        [[step, amount, newBalInitiator, oldBalRecipient]],
        columns=["step", "amount", "newBalInitiator", "oldBalRecipient"]
    )
    
    prediction = model.predict(X_new)[0]
    label = "Fraud" if prediction == -1 else "Not Fraud"
    
    return int(prediction), label

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Step"),
        gr.Number(label="Amount"),
        gr.Number(label="New Balance Initiator"),
        gr.Number(label="Old Balance Recipient")
    ],
    outputs=[
        gr.Number(label="Prediction (0=Normal, 1=Fraud)"),
        gr.Label(label="Result")
    ],
    title="Fraud Detection with Isolation Forest",
    description="Enter transaction details and the model will predict whether it's fraud (1) or not (0)."
)

if __name__ == "__main__":
    iface.launch()
