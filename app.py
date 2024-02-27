import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template

def get_pdf_text_content(pdf_storage):
    
    ## Variable that will store the raw text from pdf(s).
    raw_text = ""; ## Initialized

    ## Loop over the pdf objects and read them and append it to text.
    for text in pdf_storage:
        ## Creates a pdf object that has pages
        pdf_reader = PdfReader(text)

        ## Loop over the pages to read content on each page
        for page in pdf_reader.pages:
            ## Extracts raw text from the page of the pdf.
            raw_text += page.extract_text()
    return(raw_text)

def get_text_chunks(raw_text):
    ## chunksize = number of characters, chunk_overlap makes sure to start the next chunk 200 characters before to keep the meaning intact of 
    ## sentence.
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size = 1000, chunk_overlap = 200, length_function = len);
    chunks = text_splitter.split_text(raw_text);
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl");
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding=embeddings);
    return(vectorstore)

def get_conversation_chain(vectorstore):
    llm_model = HuggingFaceHub(repo_id = "google/flan-t5-xxl", model_kwargs = {'temperature': 0.2, 'max_length':512})
    chain_memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True);
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm_model, 
        retriever= vectorstore.as_retriever(),
        memory = chain_memory)
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():

    ## To access API Keys using Langchain
    load_dotenv()
    ## Page configuration
    st.set_page_config(page_title = "ChatBot for PDFs", page_icon=":book:")

    st.write(css, unsafe_allow_html=True);

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    ## Page header
    st.header("Chat with your PDF :book:")

    ## Text input
    user_question = st.text_input("Ask a question about PDF:");

    if user_question:
        handle_userinput(user_question)

    #st.write(user_template.replace("{{MSG}}", "Hello chatbot"), unsafe_allow_html=True)
    #st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True);

    ## Sidebar for uploading document
    with st.sidebar:
        ##Contents of sidebar
        ## Subheader
        st.subheader("Document Uploader")
        pdf_storage = st.file_uploader("Upload PDF & click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            ## Program is running
            with st.spinner("Processing PDF"):

                ## get pdf text
                ### This will return a single string of text content from the PDF(s).
                raw_text = get_pdf_text_content(pdf_storage);
                #st.write(raw_text);

                ## break it into text chunks
                ### Returns a list of chunks of text that will be stored in the db.
                text_chunks = get_text_chunks(raw_text);
                #st.write(text_chunks)

                ## creates vectorstore and embeddings
                ### Creates vector representation of chunks of text and stores it in vectordb
                vectorstore = get_vectorstore(text_chunks)

                ## Conversational chain - Adds memory to the chat bot. (Takes history and answers the next question accordingly)
                st.session_state.conversation = get_conversation_chain(vectorstore); # linked to the that session


if __name__ == "__main__":
    main()
