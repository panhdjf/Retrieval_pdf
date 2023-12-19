import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from htmlTemplates import css, bot_template, user_template

from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import torch
from transformers import  AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer, pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
import time

import langchain
langchain.verbose = True
from PyPDF2 import PdfReader

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(docs):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(docs)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert", model_kwargs={"device": device})
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore





def get_conversation_chain(vectorstore):
    # SYSTEM_PROMPT = "Sử dụng các phần ngữ cảnh sau đây để trả lời câu hỏi ở cuối. Nếu không biết câu trả lời, bạn chỉ cần nói rằng bạn không biết."

    # def generate_prompt(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    #     return f"""
    # ### Câu hỏi:

    # {system_prompt}
    # {prompt}


    # ### Trả lời:
    # """.strip()

    # template = generate_prompt( 
    # """
    # {context}

    # Câu hỏi: {question}
    # """,
    #     system_prompt=SYSTEM_PROMPT,
    # )


    # prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    template = """
        Sử dụng các phần ngữ cảnh sau đây (được phân cách bởi <ctx></ctx>) để trả lời câu hỏi ở cuối:
        ------
        <ctx>
        {context}
        </ctx>
        ------
        {question}
        Answer:
        """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    condense_prompt = PromptTemplate.from_template(
        ('Do X with user input ({question}), and do Y with chat history ({chat_history}).')
    )

    # combine_docs_custom_prompt = PromptTemplate.from_template(
    #     ('Write a haiku about a dolphin.\n\n'
    #     'Completely ignore any context, such as {context}, or the question ({question}).')
    # )

    '''---------Loading model VNAI-llama2----------'''
    print('Loading model VNAI-llama2 - CPU')
    model_name_or_path = "VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain"
    cache_dir = '/home/vinbig/Documents/PA_Modeling/Retrieval_pdf/tmp1'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
        pretraining_tp=1,
        cache_dir=cache_dir,
    )

    '''---------Loaded Model---------'''

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.,
        top_p=0.95,
        repetition_penalty=1.15,
        streamer=streamer,
    )

    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0., "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True,
        # prompt=prompt,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs=dict(prompt=prompt)

    )

    return conversation_chain



def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                start = time.time()
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                print("Thời gian: ", time.time() - start) 


if __name__ == '__main__':
    main()