import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env (especially OpenAI API key)

# Set the title for the Streamlit app
st.title("URL DATA INSIGHT ENGINE 📈")

# Create a sidebar with a title for user input
st.sidebar.title("Article links")

# Create an empty list to store user-entered URLs
urls = []
url_count = 2  # Initial number of URL input fields

# Allow users to input article URLs
for i in range(url_count):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

# Button to add more URL input fields
if st.sidebar.button("Add More URLs"):
    url_count += 1
    urls.append("")  # Add an empty input field

# Create a button to trigger URL processing
process_url_clicked = st.sidebar.button("Process URLs")

# Define a file path for storing the FAISS index
file_path = "faiss_store_openai.pkl"

# Create a placeholder for dynamic content
main_placeholder = st.empty()

# Initialize the OpenAI language model
llm = OpenAI(temperature=0.9, max_tokens=500)

# Check if the "Process URLs" button has been clicked
if process_url_clicked:
    # Load data from the specified URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split the data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and save them to a FAISS index
    embeddings = OpenAIEmbeddings(openai_api_key="OPENAI_API_KEY")
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file         
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

# Allow the user to input a question
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }

            # Display the answer to the user
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
