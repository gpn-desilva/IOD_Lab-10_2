import streamlit as st
import pandas as pd
import joblib
from utils import preprocessor

def run():
    # ✅ Load the trained pipeline
    model = joblib.load("model.joblib")

    st.title("Sentiment Analysis")
    st.text("Basic app to detect the sentiment of text.")
    st.text("")
    
    # ✅ User input box
    userinput = st.text_input('Enter text below, then click the Predict button.', placeholder='Input text HERE')
    st.text("")
    
    predicted_sentiment = ""
    
    # ✅ Prediction logic
    if st.button("Predict"):
        input_series = pd.Series([userinput])
        predicted_sentiment = model.predict(input_series)[0]

        if predicted_sentiment == 1:
            output = 'positive 👍'
        else:
            output = 'negative 👎'

        sentiment = f'Predicted sentiment of "{userinput}" is {output}.'
        st.success(sentiment)

# ✅ Entry point
if __name__ == "__main__":
    run()
