import streamlit as st
from mymodel import *
from utils import *

# Initialize your HateDetector model
HateDetector = CustomModel('models/binary', 'models/multi')

# Function to analyze sentiment
def analyze_sentiment(user_input):
    if not user_input:
        return user_input, "Please enter a sentence."
    
    logits_list = HateDetector.inference(user_input)
    probabilities_list = softmax_grouped(logits_list) 
    return probabilities_list

def main():
    st.title('Hate Speech Detection')
    
    user_input = st.text_input('Enter a sentence:')
    if st.button('Analyze'):
        probabilities = analyze_sentiment(user_input)
        labels = ["Non-Hate", "Hate", "Acceptable", "Inappropriate", "Offensive", "Violent"]
        st.write("Sentiment Probabilities:")
        for label, probability in zip(labels, probabilities):
            st.write(f"{label}: {probability:.2f}")
    
if __name__ == '__main__':
    main()





