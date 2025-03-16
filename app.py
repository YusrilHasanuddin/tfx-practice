import streamlit as st

from prediction import predict

# Define your server URL
SERVER_URL = "http://localhost:8500/v1/models/political-bias-detection-model:predict"


# Streamlit app
def main():
    st.title("Political Bias Detection")
    st.write("Enter text to analyze its political bias.")

    user_input = st.text_area("Input Text")

    if st.button("Predict"):
        if user_input:
            try:
                prediction_result = predict(SERVER_URL, user_input)
                st.write(f"Prediction: {prediction_result}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text for prediction.")


if __name__ == "__main__":
    main()
