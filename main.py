# from langchain.document_loaders import PyPDFDirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import  HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS

# import torch
# from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# from langchain import HuggingFacePipeline, PromptTemplate
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
# from os import listdir
# from os.path import isfile, join
# from transformers import pipeline
# from transformers import TextStreamer, pipeline

# import ipywidgets as widgets
# from IPython.display import display, HTML

# device = torch.device("cpu")

# # Read pdf
# loader = PyPDFDirectoryLoader("pdfs")
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
# texts = text_splitter.split_documents(docs)
# # texts = texts[:1]

# def get_text_chunks(texts):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = []
#     for text in texts:
#         chunks += text_splitter.split_text(text.page_content)
#     return chunks


# def get_vectorstore(text_chunks):
#     # embeddings = OpenAIEmbeddings()
#     embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": device})
#     # embeddings = HuggingFaceInstructEmbeddings(
#     # model_name="hkunlp/instructor-large", model_kwargs={"device": device})
    

#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore

# text_chunks = get_text_chunks(texts)
# vectorstore = get_vectorstore(text_chunks)


# HUGGINGFACEHUB_API_TOKEN = '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'
# """# PhoGPT"""
# model_path = "vinai/PhoGPT-7B5-Instruct"
# config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, use_auth_token=HUGGINGFACEHUB_API_TOKEN)
# config.init_device = "cpu"
# # config.attn_config['attn_impl'] = 'triton' # Enable if "triton" installed!

# # model = AutoModelForCausalLM.from_pretrained(
#     # model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True, use_auth_token=HUGGINGFACEHUB_API_TOKEN)
# # If your GPU does not support bfloat16:
# model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True, use_auth_token=HUGGINGFACEHUB_API_TOKEN)
# model.eval()

# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,  use_auth_token=HUGGINGFACEHUB_API_TOKEN)


# DEFAULT_SYSTEM_PROMPT = """
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
# """.strip()


# def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
#     return f"""
# [INST] <>
# {system_prompt}
# <>

# {prompt} [/INST]
# """.strip()

# streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# text_pipeline = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=1024,
#     temperature=0,
#     top_p=0.95,
#     repetition_penalty=1.15,
#     streamer=streamer,
# )


# SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

# template = generate_prompt(
#     """
# {context}

# Question: {question}
# """,
#     system_prompt=SYSTEM_PROMPT,
# )


# prompt = PromptTemplate(template=template, input_variables=["context", "question"])




# def get_conversation_chain(vectorstore):
#     # llm = ChatOpenAI()
#     llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

#     memory = ConversationBufferMemory(
#         memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain




# def display_message(content, is_user=True):
#     if is_user:
#         display(HTML(user_template.replace("{{MSG}}", content)))
#     else:
#         display(HTML(bot_template.replace("{{MSG}}", content)))

# # Thực hiện lặp trò chuyện và hiển thị tin nhắn
# def display_chat_history(chat_history):
#     for i, message in enumerate(chat_history):
#         is_user = i % 2 == 0
#         display_message(message.content, is_user)



# user_question_input = widgets.Text(
#     value='',
#     description='Ask a question about your documents:',
# )

# # Tạo nút để xử lý câu hỏi
# process_button = widgets.Button(description="Process")

# # Tạo ô hiển thị lịch sử trò chuyện
# output_area = widgets.Output()


# def on_process_button_click(b):
#     with output_area:
#         # raw_text = get_pdf_text(pdf_docs)
#         # text_chunks = get_text_chunks(raw_text)
#         # vectorstore = get_vectorstore(text_chunks)
#         conversation = get_conversation_chain(vectorstore)

#         # Giả định user_question_input là một widget Text
#         user_question = user_question_input.value
#         response = conversation({'question': user_question})
#         chat_history = response['chat_history']

#         display_chat_history(chat_history)

# # Gắn sự kiện click cho nút
# process_button.on_click(on_process_button_click)

# # Hiển thị các widget
# display(user_question_input)
# display(process_button)
# display(output_area)



from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import  HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
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
from sentence_transformers import SentenceTransformer
import langchain
langchain.verbose = True


# device = torch.device("cpu")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# Read pdf
loader = PyPDFDirectoryLoader("pdfs")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
texts = text_splitter.split_documents(docs)

embeddings_vi_2 = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert", model_kwargs={"device": device})
print('--------Embedding-------')

vectorstore = FAISS.from_documents(texts, embedding=embeddings_vi_2)

print('---------------Loading Model--------------')

HUGGINGFACEHUB_API_TOKEN = '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'

# '''-----------------'''
# """# PhoGPT"""
# model_path = "vinai/PhoGPT-7B5-Instruct"
# config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, use_auth_token=HUGGINGFACEHUB_API_TOKEN)
# config.init_device = "cpu"
# # config.attn_config['attn_impl'] = 'triton' # Enable if "triton" installed!
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,  use_auth_token=HUGGINGFACEHUB_API_TOKEN)

# model = AutoModelForCausalLM.from_pretrained(
#     model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True, use_auth_token=HUGGINGFACEHUB_API_TOKEN)
# # If your GPU does not support bfloat16:
# # model = AutoModelForCausalLM.from_pretrained(
# #     model_path,
# #     config=config,
# #     trust_remote_code=True,
# #     quantization_config=nf4_config,
# # )

# model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float32, trust_remote_code=True, use_auth_token=HUGGINGFACEHUB_API_TOKEN)
# # model.eval()


'''-----------------'''
# # BKAI_lama2
# print('Loading model Bkai-llama2')
# model_name_or_path = "bkai-foundation-models/vietnamese-llama2-7b-40GB"

# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, use_auth_token=HUGGINGFACEHUB_API_TOKEN)

# model = AutoModelForCausalLM.from_pretrained(model_name_or_path, pretraining_tp=1, torch_dtype=torch.bfloat16, use_auth_token=HUGGINGFACEHUB_API_TOKEN)
# # model = AutoModelForCausalLM.from_pretrained(
# #     model_name_or_path,
# #     # config=config,
# #     trust_remote_code=True,
# #     quantization_config=nf4_config,
# #     use_auth_token=HUGGINGFACEHUB_API_TOKEN
# # )

'''-----------------'''
print('Loading model VNAI-llama2')

print('Loading model VNAI-llama2 - CPU')
# cache_dir = '/home/vinbig/Documents/PA_Modeling/Retrieval_pdf/tmp1'
# model_name_or_path = "VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain"
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path,
#     torch_dtype=torch.float32,
#     pretraining_tp=1,
#     cache_dir=cache_dir
# )

model_name_or_path = "VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain"
cache_dir = "/home/vinbig/Documents/PA_Modeling/Retrieval_pdf/tmp"
# tokenizer = AutoTokenizer.from_pretrained(
#     model_name_or_path, 
#     cache_dir=cache_dir,
#     padding_side="right",
#     use_fast=True, # Fast tokenizer giving issues.
#     tokenizer_type='llama', #if 'llama' in args.model_name_or_path else None, # Needed for HF name change
#     use_auth_token=True, 
# )
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True,cache_dir=cache_dir,)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    # load_in_8bit=True,
    torch_dtype=torch.float32,
    pretraining_tp=1,
    # use_auth_token=True,
    # trust_remote_code=True,
    cache_dir=cache_dir,
)

print('---------Loaded Model---------')



SYSTEM_PROMPT = "Sử dụng các phần ngữ cảnh sau đây để trả lời câu hỏi ở cuối. Nếu không biết câu trả lời, bạn chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời."

def generate_prompt(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    return f"""
### Câu hỏi:

{system_prompt}
{prompt}


### Trả lời:
""".strip()

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

text_pipeline = pipeline(
    "text-generation",
    model=model.to('cpu'),
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.,
    top_p=0.95,
    repetition_penalty=1.15,
    streamer=streamer,
)

template = generate_prompt(
    """
{context}

Câu hỏi: {question}
""",
    system_prompt=SYSTEM_PROMPT,
)


prompt = PromptTemplate(template=template, input_variables=["context", "question"])


qa_chain = RetrievalQA.from_chain_type(
    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.}),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    verbose=True,
)

import time

while True: 
    m = int(input("Nhập một số: ") )
    if m == 0: 
        break 
    
    ques = input("Nhập câu hỏi: ")
    start = time.time()
    result = qa_chain(ques)
    time_gen = time.time() - start
    print("Time:", time_gen)
