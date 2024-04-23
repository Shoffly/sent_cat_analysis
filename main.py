import streamlit as st
import pandas as pd
from openai import OpenAI
import base64
import os


key = st.secrets["openai"]["api_key"]
# Initialize OpenAI API client with your API key
client = OpenAI(key)

# Sample CSV data
sample_data = """Review
"Not available at all branches"
"App glitch - i canceled the order before the timer finished and i got SMS that the order confirmed ."
"Even i order from the app or the counter i will get my beans"
"Simplicity and the steps written and clear ."
"Perfect and easy to use"
"Simplicity"
"The rewards is beneficial."
"Simplicity"
"So nice and easy"
"Creative i love it."
"The rewards is beneficial."
"""


# Function to allow download of sample CSV file
def download_sample_csv(data):
    csv = data.encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sample_data.csv">Download sample CSV file</a>'
    return href

# Streamlit app
def main():
    st.title("Reviews Analysis Tool")

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    # Add download link for sample CSV file
    st.markdown(download_sample_csv(sample_data), unsafe_allow_html=True)
    st.write("This is a sample of how the reviews should look like.")


    if uploaded_file is not None:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Choose column with reviews
        review_column = st.selectbox("Select review column", df.columns)

        # Define buckets dynamically based on user input
        buckets_input = st.text_input("Enter buckets (comma-separated)")
        buckets = buckets_input.split(',')

        if st.button("Analyze"):
            # Send data and buckets to OpenAI API
            response = analyze_data(df[review_column], buckets)

            # Update DataFrame with categorization and sentiment columns
            df['Category'] = response['categories']
            df['Sentiment'] = response['sentiments']

            # Display updated DataFrame
            st.write(df)

            # Allow downloading processed DataFrame as CSV
            st.download_button(label="Download CSV", data=df.to_csv(), file_name="processed_data.csv", mime="text/csv")


# Function to analyze data using OpenAI API
def analyze_data(data, buckets):
    categories = []
    sentiments = []

    for review in data:
        # Construct messages list dynamically
        messages = [
            {
                "role": "system",
                "content": f"You will be provided with a review and categories the reviews into these buckets: {', '.join(buckets)} and classify its sentiment as positive, neutral, or negative."
            },
            {
                "role": "user",
                "content": review
            }
        ]

        # Call OpenAI API for chat completion
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        print(response)
        # Extracting the message content
        content = response.choices[0].message.content

        # Splitting the content into lines
        content_lines = content.split('\n')

        # Parsing classification and sentiment
        classification = None
        sentiment = None

        for line in content_lines:
            if line.startswith("Category:"):
                classification = line.split(": ")[1]

            elif line.startswith("Sentiment:"):
                sentiment = line.split(": ")[1]

        categories.append(classification)
        print(classification)
        sentiments.append(sentiment)
        print(sentiments)
    return {'categories': categories, 'sentiments': sentiments}


if __name__ == "__main__":
    main()
