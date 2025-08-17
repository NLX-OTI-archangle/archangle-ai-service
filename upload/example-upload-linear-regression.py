import gradio as gr
import joblib
import numpy as np

model = joblib.load("models/linear_regression_model.pkl")

def predict(x):
    x_arr = np.array(x).reshape(-1, 1)
    prediction = model.predict(x_arr)
    return float(prediction[0])

iface = gr.Interface(
    fn=predict,
    inputs=gr.Number(label="Input X"),
    outputs=gr.Number(label="Predicted Y"),
    title="Linear Regression Demo",
    description="A simple linear regression with StandardScaler + LinearRegression pipeline"
)

if __name__ == "__main__":
    iface.launch()
