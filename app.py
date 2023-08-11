import streamlit as st
import os
import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms import HuggingFacePipeline

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

@st.cache_resource()
def LLMPipeline(temperature, 
                top_p,
                top_k,
                max_length,
                repetition_penalty=1.1,
                model_id='meta-llama/Llama-2-7b-chat-hf',
                hf_auth='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'):
    
    # Initialize tokenizer & model
    tokenizer = LlamaTokenizer.from_pretrained(model_id, token=hf_auth)
    model = LlamaForCausalLM.from_pretrained(model_id, token=hf_auth)
    model.eval()
    
    # Define HF pipeline
    generate_text = pipeline(model=model,
                             tokenizer=tokenizer,
                             return_full_text=True,
                             task='text-generation',
                             temperature=temperature
                             top_p=top_p,
                             top_k=top_k,                         
                             max_new_tokens=max_length,
                             repetition_penalty=repetition_penalty)
    
    llm = HuggingFacePipeline(pipeline=generate_text)
  
    return llm

@st.cache_resource()
def get_conversation(llm, window_len=5):
    # Define memory
    window_memory = ConversationBufferWindowMemory(k=window_len)
    conversation = ConversationChain(
        llm=llm, 
        verbose=False, 
        memory=window_memory
    )

    conversation.prompt.template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. You are the AI, and your role is to answer questions respectfully. Current conversation:\n{history}\nHuman: {input}\nAI:"""
    
    return conversation

# Function for generating LLaMA2 response
def generate_llama2_response(conversation, prompt_input):
#     string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
#     for dict_message in st.session_state.messages:
#         if dict_message["role"] == "user":
#             string_dialogue += "User: " + dict_message["content"] + "\\n\\n"
#         else:
#             string_dialogue += "Assistant: " + dict_message["content"] + "\\n\\n"
            
#     output = chat_model(f"prompt {string_dialogue} {prompt_input} Assistant: ")
    
    return conversation.predict(input=prompt_input)

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')    
    # Text generation params
    st.subheader('Text generation parameters')
    temperature = st.sidebar.slider('temperature', min_value=0.1, max_value=1.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    top_k = st.sidebar.slider('top_k', min_value=0, max_value=100, value=20, step=10)
    max_length = st.sidebar.slider('max_length', min_value=64, max_value=4096, value=512, step=8)
    
    # Load llm & conversation
    llm = LLMPipeline(temperature, top_p, top_k, max_length)
    conversation = get_conversation(llm)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

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
        with st.spinner("Thinking..."):
            response = generate_llama2_response(conversation, prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
