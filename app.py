import streamlit as st
from datetime import datetime
from main import ask  # Import the ask function from main.py
from PIL import Image  # Import to handle local image loading
import os
import pandas as pd
from main import url_cache

# Load the local image
image_path = './assets/indago.PNG'   # Update the path to your image
logo_image = Image.open(image_path)  # Use PIL to open the image

# Store the chat history and FAISS cache in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'faiss_cache' not in st.session_state:
    st.session_state.faiss_cache = {}  # Cache to store FAISS indexes for each URL

# Add a logo at the top
st.sidebar.image(logo_image, width=200)  # Display the local image

# Center the title
st.markdown("<h1 style='text-align: center;'>Indago Query Intelligence</h1>", unsafe_allow_html=True)

# Sidebar Chat history section (Automatically generated titles)
st.sidebar.subheader("Chat History")

# Add a button to delete chat history
if st.sidebar.button("Delete Chat History"):
    st.session_state.chat_history = []  # Clear the chat history
    st.success("Chat history deleted!")

# Add a button to clear FAISS cache and URL cache
if st.sidebar.button("Clear Cache"):
    # Clear FAISS cache
    st.session_state.faiss_cache = {}
    # Clear URL cache in main.py (if global or passed state is being used)
    from main import url_cache  # Make sure url_cache is accessible from main.py
    url_cache.clear()  # Assuming url_cache is a global variable in main.py

    st.success("Cache cleared successfully!")

# Display the chat history if available
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.sidebar.write(f"**Title:** {chat['title']}")
        st.sidebar.write("---")
else:
    st.sidebar.write("No chat history yet.")

# Main page form to enter URLs and query
with st.form("url_query_form"):
    # Text area for users to input URLs explicitly, without any predefined values
    urls_input = st.text_area("Submit the Access Points", placeholder="Your Links here")
    query = st.text_input("Got a Question?", placeholder="Query here")
    submit_button = st.form_submit_button("Get Response")

# Process user input when the form is submitted
if submit_button:
    if not urls_input.strip():
        st.error("Please enter at least one URL.")
    elif not query.strip():
        st.error("Please enter a query.")
    else:
        # Process the input if both URLs and query are provided
        urls = [url.strip() for url in urls_input.split("\n") if url.strip()]

        # Paths to be scraped
        paths = [
            "/products",
            "/services",
            "/blog",
            "/testimonials",
            "/reviews"
        ]

        # Show a spinner with an emoji while the response is being generated
        with st.spinner("⚙️ Generating response, please wait..."):
            # Call the 'ask' function with the user inputs and use cache for FAISS indexes
            response = ask(query, urls, paths, st.session_state.faiss_cache)  # Pass the cache as an argument

        # If the response is a list, join the responses into a single string separated by double newlines
        if isinstance(response, list):
            response = "\n\n".join(response)

        # Replace newlines with <br> for HTML rendering in Streamlit
        response_with_html = response.replace("\n", "<br>")

        # Generate a title using timestamp or count for history
        title = f"Entry at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        st.session_state.chat_history.append({"title": title, "response": response_with_html})

        # Display the response on the main page using markdown and allowing HTML for line breaks
        st.success(f"🎉 Query added to chat history with title: {title}")
        st.markdown(f"Response: {response_with_html}", unsafe_allow_html=True)

        # **Save responses to a spreadsheet**
        response_list = response.split("\n\n")  # Assuming double newlines separate questions/answers

        # Check if the output file already exists
        output_file_path = os.path.join(os.getcwd(), "output_responses.xlsx")

        if os.path.exists(output_file_path):
            # If the file exists, load the existing data
            existing_df = pd.read_excel(output_file_path)
        else:
            # If the file doesn't exist, create an empty DataFrame
            existing_df = pd.DataFrame(columns=["URL", "Questions", "Answers"])

        # Prepare the new data for appending
        new_data = {"URL": [], "Questions": [], "Answers": []}

        # Loop through URLs and associate them with questions and answers
        for item in response_list:
            url = None
            question = None
            answer = None

            # Split each item into lines and parse the URL, Query, and Answer
            for line in item.split("\n"):
                if line.startswith("URL:"):
                    url = line.replace("URL:", "").strip()  # Extract and clean URL
                elif line.startswith("Query:"):
                    question = line.replace("Query:", "").strip()  # Extract and clean Query
                else:
                    answer = line.strip()  # Everything else is the answer

            # Append the extracted information to the new_data dictionary
            if url and question and answer:
                new_data["URL"].append(url)
                new_data["Questions"].append(question)
                new_data["Answers"].append(answer)

        # Create a new DataFrame for the new data
        new_df = pd.DataFrame(new_data)

        # Append the new data to the existing DataFrame
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)

        # Save the updated DataFrame to the same Excel file
        updated_df.to_excel(output_file_path, index=False)

        st.success(f"Responses appended and saved to {output_file_path}")
