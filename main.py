from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import  HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from langchain import HuggingFacePipeline, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from os import listdir
from os.path import isfile, join
from transformers import pipeline
from transformers import TextStreamer, pipeline

import ipywidgets as widgets
from IPython.display import display, HTML

device = torch.device("cpu")

# Read pdf
loader = PyPDFDirectoryLoader("pdfs")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
texts = text_splitter.split_documents(docs)
# texts = texts[:1]

def get_text_chunks(texts):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = []
    for text in texts:
        chunks += text_splitter.split_text(text.page_content)
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": device})
    # embeddings = HuggingFaceInstructEmbeddings(
    # model_name="hkunlp/instructor-large", model_kwargs={"device": device})
    

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

text_chunks = get_text_chunks(texts)
vectorstore = get_vectorstore(text_chunks)


HUGGINGFACEHUB_API_TOKEN = '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'
"""# PhoGPT"""
model_path = "vinai/PhoGPT-7B5-Instruct"
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, use_auth_token=HUGGINGFACEHUB_API_TOKEN)
config.init_device = "cpu"
# config.attn_config['attn_impl'] = 'triton' # Enable if "triton" installed!

# model = AutoModelForCausalLM.from_pretrained(
    # model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True, use_auth_token=HUGGINGFACEHUB_API_TOKEN)
# If your GPU does not support bfloat16:
model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True, use_auth_token=HUGGINGFACEHUB_API_TOKEN)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,  use_auth_token=HUGGINGFACEHUB_API_TOKEN)


DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()


def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <>
{system_prompt}
<>

{prompt} [/INST]
""".strip()

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15,
    streamer=streamer,
)


SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

template = generate_prompt(
    """
{context}

Question: {question}
""",
    system_prompt=SYSTEM_PROMPT,
)


prompt = PromptTemplate(template=template, input_variables=["context", "question"])




def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain




def display_message(content, is_user=True):
    if is_user:
        display(HTML(user_template.replace("{{MSG}}", content)))
    else:
        display(HTML(bot_template.replace("{{MSG}}", content)))

# Thực hiện lặp trò chuyện và hiển thị tin nhắn
def display_chat_history(chat_history):
    for i, message in enumerate(chat_history):
        is_user = i % 2 == 0
        display_message(message.content, is_user)



user_question_input = widgets.Text(
    value='',
    description='Ask a question about your documents:',
)

# Tạo nút để xử lý câu hỏi
process_button = widgets.Button(description="Process")

# Tạo ô hiển thị lịch sử trò chuyện
output_area = widgets.Output()


def on_process_button_click(b):
    with output_area:
        # raw_text = get_pdf_text(pdf_docs)
        # text_chunks = get_text_chunks(raw_text)
        # vectorstore = get_vectorstore(text_chunks)
        conversation = get_conversation_chain(vectorstore)

        # Giả định user_question_input là một widget Text
        user_question = user_question_input.value
        response = conversation({'question': user_question})
        chat_history = response['chat_history']

        display_chat_history(chat_history)

# Gắn sự kiện click cho nút
process_button.on_click(on_process_button_click)

# Hiển thị các widget
display(user_question_input)
display(process_button)
display(output_area)