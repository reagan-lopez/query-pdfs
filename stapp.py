import os
import logging
import shutil
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import streamlit as st

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store_name = "vector_store"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Delete vector store
def del_vector_store():
    try:
        shutil.rmtree(vector_store_name)
        logger.info(f"The directory '{vector_store_name}' has been deleted.")
    except Exception as e:
        logger.exception(f"Error deleting the directory: {e}")


# Extract text from pdf.
def get_pdf_text(pdfs):
    pdf_text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    return pdf_text


# Split text into chunks.
def get_text_chunks(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    text_chunks = text_splitter.split_text(pdf_text)
    return text_chunks


# Embed the chunks and insert into vector store.
def set_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(vector_store_name)


# Define chain using model and prompt template.
def get_chain():
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "Sorry, I'm unable to fetch that information from the documents.", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# Generate response using the vector store and model.
def get_response(user_query):
    try:
        vector_store = FAISS.load_local(vector_store_name, embeddings=embeddings)
    except Exception as e:
        return "Please upload PDFs to be queried."

    else:
        docs = vector_store.similarity_search(user_query)
        chain = get_chain()
        response = chain(
            {"input_documents": docs, "question": user_query}, return_only_outputs=True
        )
        return response["output_text"]


# Format the message to be stored in the streamlit session state.
def get_message(role, content):
    return {"role": role, "content": content}


if __name__ == "__main__":
    st.set_page_config("Query PDFs")
    st.header("Query PDFs using GeminiPro")

    # Delete vector store on first session run
    if "first_run" not in st.session_state:
        if os.path.exists(vector_store_name):
            del_vector_store()
        st.session_state["first_run"] = True

    # Upload PDF section
    with st.sidebar:
        st.title("Menu:")
        pdfs = st.file_uploader("Upload PDFs:", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing.."):
                pdf_text = get_pdf_text(pdfs)
                text_chunks = get_text_chunks(pdf_text)
                set_vector_store(text_chunks)
                st.success("Done!")

    # Initialize "messages" key in streamlit session
    if "messages" not in st.session_state.keys():
        message = get_message("assistant", "Upload PDFs and query.")
        st.session_state.messages = [message]

    # User prompt
    if user_prompt := st.chat_input():
        message = get_message("user", user_prompt)
        st.session_state.messages.append(message)

    # Display all messages in current session
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Generate last message
    last_message = st.session_state.messages[-1]
    if last_message["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Processing.."):
                response = get_response(user_prompt)
                st.write(response)
                message = get_message("assistant", response)
                st.session_state.messages.append(message)
