import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate

OpenAI_API_KEY=" "#add api key 

st.header("YOURBOT")

with st.sidebar:
    st.title("My Notes")
    file=st.file_uploader("Upload notes pdf and start asking questions",type="pdf")

#extracting text from pdf and chunking
if file is not None:
    my_pdf=PdfReader(file)
    text=""
    for page in my_pdf.pages:
        text+=page.extract_text()
        #st.write(text)

#breaking as chunks
    splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
    chunks=splitter.split_text(text)
    st.write(chunks)

    embeddings=OpenAIEmbeddings(api_key=OpenAI_API_KEY)

    vector_store=FAISS.from_texts(chunks,embeddings)

    user_query=st.text_input("Type your query here:")

    #semantic search for checking similar words
    if user_query:
        matching_chunks=vector_store.similarity_search(user_query)

        #define our llm
        llm=ChatOpenAI(
            api_key=OpenAI_API_KEY,
            max_tokens=300,
            temperature=0,    #temp is for randomness
            model="gpt-3.5-turbo"
        )

        #asking qns
        chain=load_qa_chain(llm,chain_type="stuff")
        output=chain.run(question=user_query,input_documents=matching_chunks)
        st.write(output)

        customized_prompt=ChatPromptTemplate.from_template(
           """You are my assistant tutor.Answer the question based on the following context and 
           if you did not get the context simply say "Sorry!! I don't know ,yet to train":
           {context}
           Question:{input}
           """
        )

        chain=create_stuff_documents_chain(llm,customized_prompt)
        output=chain.invoke({"input":user_query,"context":matching_chunks})
        st.write(output)