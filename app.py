#!pip install streamlit

import pickle

# Assuming you have trained a model named 'model'
with open('cyberbully_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

if 'error_text' not in st.session_state:
    st.session_state.error_text = ''

if 'prediction' not in st.session_state:
    st.session_state.prediction = False

if 'mydataframe' not in st.session_state:
    st.session_state.mydataframe = pd.DataFrame()

def handle_prediction():
    try:
        input_text = st.session_state['input_text']
        st.session_state.error_text = ''
    except ValueError:
        st.session_state.error_text = "Invalid input."
        return

    # Transform the input text using the vectorizer
    input_vectorized = vectorizer.transform([input_text])

    # Predict using the loaded model
    prediction = model.predict(input_vectorized)[0]

    # Create a DataFrame with the input text and prediction
    df = pd.DataFrame({
        'datetime': [datetime.now().strftime("%d-%b-%y %H:%M:%S")],
        'text': [input_text],
        'prediction': [prediction]
    })

    # Update the session state dataframe
    if len(st.session_state.mydataframe) == 0:
        st.session_state.mydataframe = df
    else:
        st.session_state.mydataframe = pd.concat([st.session_state.mydataframe, df], ignore_index=True)

    st.rerun()

st.title("Cyberbully Prediction System")

# Input text area
st.text_area("Enter text to analyze for cyberbullying:", key='input_text')

# Prediction button
if st.button('Predict'):
    handle_prediction()

# Display error messages if any
st.write(st.session_state.error_text)

# Display total predictions
st.write(f"Total Texts Analyzed: {len(st.session_state.mydataframe)}")

# Display the dataframe with the results
if len(st.session_state.mydataframe) > 0:
    st.dataframe(st.session_state.mydataframe)
