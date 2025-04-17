from transformers import pipeline, TFAutoModelForSeq2SeqLM, AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks import LangChainTracer
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter #This liberary is used to create Chunks of data.
from langchain_core.output_parsers import StrOutputParser  #This liberary is used to provide output in string.
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import PyPDFLoader  #This liberary is used to load Pdf documents.
from langchain.document_loaders import PyPDFDirectoryLoader # This liberary is used to fetch text from pdf.
from langchain_core.prompts import ChatPromptTemplate  #This liberary is used to create Prompts.
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.llms import HuggingFacePipeline
import os #Importing the Os environment.
import streamlit as st
from dotenv import load_dotenv
from langsmith import utils

#Lagsmith is used to trace(maintaining the track) of Langchain model.

import os
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=""
HuggingFace_Key=os.getenv("HuggingFace_Key")
os.environ["LANGCHAIN_PROJECT"]="RAG_APPLICATION"
load_dotenv(dotenv_path=".env",override=True)

utils.tracing_is_enabled()


#Loading the file and Dividing the data into chunks..
@st.cache_resource
def load_and_process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks

# Creating the Embeddings. 

@st.cache_resource
def create_embeddings():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

# Create the Chromadb vector store from the chunks and embeddings
CHROMA_PERSIST_DIR = "chroma_db"

@st.cache_resource
def create_vectorstore(_chunks,_embeddings):
    if os.path.exists(CHROMA_PERSIST_DIR):
        db = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    else:
        db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PERSIST_DIR)
        db.persist()
    return db.as_retriever()




@st.cache_resource
def create_llm():
    model_name_llm = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name_llm)
    tf_model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name_llm)
    hf_pipeline = pipeline("text2text-generation", model=tf_model, tokenizer=tokenizer, framework="tf")
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return llm


# Create the RetrievalQA chain
def run_rag_chain(query, retriever, llm, tracer=None):
    from langchain.chains import RetrievalQA
    callbacks = [tracer] if tracer else None
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,  # You can change this
        callbacks=callbacks
    )
    result = qa({"query": query})
    return result


# Streamlit User Setup.
st.title("Question Answering from PDF")
pdf_file = st.file_uploader("Upload your PDF document", type="pdf")

# Initialize LangSmith Tracer (only if API key is set)
tracer = LangChainTracer() if os.getenv("LANGCHAIN_API_KEY") else None

if pdf_file is not None:
    with st.spinner("Processing document..."):
        # Save the uploaded file temporarily
        with open("temp_document.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())
        pdf_path = "temp_document.pdf"
        chunks = load_and_process_pdf(pdf_path)
        embeddings = create_embeddings()
        retriever = create_vectorstore(chunks, embeddings)
        llm = create_llm()
        st.success("Document processed and database created/loaded!")

    query = st.text_input("Ask a question about the document:")

    if query and "retriever" in locals() and "llm" in locals():
        with st.spinner("Generating answer..."):
            result = run_rag_chain(query, retriever, llm, tracer)
            st.subheader("Answer:")
            st.write(result["result"])

            if st.checkbox("Show Source Documents"):
                st.subheader("Source Documents:")
                for doc in result["source_documents"]:
                    st.write(doc.page_content)
                    st.markdown("---")

    # Clean up the temporary file
    if "pdf_path" in locals() and os.path.exists(pdf_path):
        os.remove(pdf_path)

st.sidebar.header("About")
st.sidebar.markdown("A basic RAG application using Langchain, Hugging Face, and ChromaDB.")
if os.getenv("LANGCHAIN_TRACING"):
    st.sidebar.markdown("[LangSmith Dashboard](https://smith.langchain.com/)")