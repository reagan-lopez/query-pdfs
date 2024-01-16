import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import streamlit as sl

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# Extract text from pdf
def get_pdf_text(pdfs):
    pdf_text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    return pdf_text


# Split text into chunks
def get_text_chunks(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    text_chunks = text_splitter.split_text(pdf_text)
    return text_chunks


# Embed the chunks and insert into vector store
def set_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("vector_store")


# Define chain using model and prompt template
def get_chain():
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def get_response(user_query):
    try:
        vector_store = FAISS.load_local("vector_store", embeddings=embeddings)
    except Exception as e:
        return "Vector store is empty. Please upload PDFs."

    else:
        if not vector_store:
            return "Please upload the PDFs to be queried!"

        docs = vector_store.similarity_search(user_query)
        chain = get_chain()
        response = chain(
            {"input_documents": docs, "question": user_query}, return_only_outputs=True
        )
        return response["output_text"]


if __name__ == "__main__":
    sl.set_page_config("Query PDFs")
    sl.header("Query PDFs using GeminiPro")

    # Upload PDF section
    with sl.sidebar:
        sl.title("Menu:")
        pdfs = sl.file_uploader("Upload PDFs:", accept_multiple_files=True)
        if sl.button("Upload"):
            with sl.spinner("Processing.."):
                pdf_text = get_pdf_text(pdfs)
                text_chunks = get_text_chunks(pdf_text)
                set_vector_store(text_chunks)
                sl.success("Done!")

    # User query section
    user_query = sl.text_input("Enter query")
    if user_query:
        sl.write(get_response(user_query))
    else:
        sl.write("Please enter valid query!")
