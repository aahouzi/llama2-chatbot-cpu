import streamlit as st
import os
import torch
import transformers
import time
import argparse
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms import HuggingFacePipeline


parser = argparse.ArgumentParser()
parser.add_argument("--hf_auth",
                    help='HuggingFace authentification token for getting LLaMa2',
                    required=True)

parser.add_argument("--window_len",
                    type=int,
                    help='Chat memory window length',
                    default=5)

args = parser.parse_args()


# App title
st.set_page_config(
    page_title="LLaMa2-7b Chatbot",
    page_icon="🦙",
    layout="centered",
)

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today ?"}]
    
def get_conversation(llm, window_len=args.window_len):
    # Define memory
    window_memory = ConversationBufferWindowMemory(k=window_len)
    conversation = ConversationChain(
        llm=llm, 
        verbose=False, 
        memory=window_memory
    )

    conversation.prompt.template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. You are the AI, so answer all the questions adressed to you respectfully. Current conversation:\n{history}\nHuman: {input}\nAI:"""
    
    return conversation

@st.cache_resource()
def LLMPipeline(temperature, 
                top_p,
                top_k,
                max_length,
                hf_auth,
                repetition_penalty=1.1,
                model_id='meta-llama/Llama-2-7b-chat-hf'):
    
    # Initialize tokenizer & model
    tokenizer = LlamaTokenizer.from_pretrained(model_id, token=hf_auth)
    model = LlamaForCausalLM.from_pretrained(model_id, token=hf_auth)
    model.eval()
    
    # Define HF pipeline
    generate_text = pipeline(model=model,
                             tokenizer=tokenizer,
                             return_full_text=True,
                             task='text-generation',
                             temperature=temperature,
                             top_p=top_p,
                             top_k=top_k,                         
                             max_new_tokens=max_length,
                             repetition_penalty=repetition_penalty)
    
    llm = HuggingFacePipeline(pipeline=generate_text)
    
    # Create langchain conversation
    conversation = get_conversation(llm)
  
    return conversation

# Replicate Credentials
with st.sidebar:
    st.title('🦙 LLaMa2-7b Chatbot')   
    
    # Text generation params
    st.subheader('Text generation parameters')
    temperature = st.sidebar.slider('temperature', min_value=0.1, max_value=1.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    top_k = st.sidebar.slider('top_k', min_value=0, max_value=100, value=20, step=10)
    max_length = st.sidebar.slider('max_length', min_value=64, max_value=4096, value=512, step=8)
    
    # Load conversation
    conversation = LLMPipeline(temperature, top_p, top_k, max_length, args.hf_auth)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today ?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("▌")
        response = conversation.predict(input=prompt)
        full_response = ""
        for item in response:
            full_response += item
            placeholder.markdown(full_response + "▌")
            time.sleep(0.04)
        placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
    
    
    
    
    
