import os
import streamlit as st
import psutil
import time
from huggingface_hub import HfApi
from requests.exceptions import HTTPError
from ipex_llm.transformers import AutoModelForCausalLM
import torch
from transformers import AutoTokenizer, GenerationConfig

# Define generation configuration
generation_config = GenerationConfig(use_cache=True, do_sample=False)

# Set page configuration
st.set_page_config(page_title="Intel GPU LLM Chatbot", page_icon=":robot_face:", layout="wide")
st.title("Intel GPU LLM Chatbot")

# Define cache directory
CACHE_DIR = os.path.abspath("./LLM_RAG_Bot/models")
os.makedirs(CACHE_DIR, exist_ok=True)

# Available models
default_models = {
    "Qwen 8B": "Qwen/Qwen-1_8B-Chat",
    "Microsoft Phi-3 mini 4K": "microsoft/Phi-3-mini-4k-instruct",
    "Microsoft Phi-3 mini 128K": "microsoft/Phi-3-mini-128k-instruct",
    "Microsoft Phi-3 small 8K": "microsoft/Phi-3-small-8k-instruct",
    "Microsoft Phi-3 small 128K": "microsoft/Phi-3-small-128k-instruct",
    "Google Gemma": "google/gemma-7b",
    "Mistral": "mistral/mistral-7b",
    "Meta LLaMA3 8B": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Meta LLaMA3 70B": "meta-llama/Meta-Llama-3-70B-Instruct"
}

# Check if the model size is less than or equal to the system RAM size
def check_model_size(model_id, access_token=None):
    api = HfApi()
    try:
        model_info = api.model_info(model_id, use_auth_token=access_token)
    except HTTPError as e:
        st.error(f"Failed to access the model. Please check your token permissions: {e}")
        return False
    
    model_size_bytes = sum(file.size for file in model_info.siblings if file.size is not None)
    model_size_gb = model_size_bytes / (1024 ** 3)
    
    system_memory = psutil.virtual_memory().total / (1024 ** 3)
    return model_size_gb <= system_memory

# Function to download the model
def download_model(model_id, access_code=None):
    if not check_model_size(model_id, access_code):
        st.error("The model size exceeds the available system RAM or there was an error accessing the model.")
        return None, None
    
    start_load_time = time.time()
    try:
        if access_code:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=access_code, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=access_code, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        end_load_time = time.time()
        load_time = end_load_time - start_load_time
        st.session_state.load_time = load_time
        st.session_state.model_path = model_id
        return tokenizer, model
    except ImportError as e:
        if 'triton' in str(e):
            st.error("This model requires the 'triton' package, which could not be found in your environment. This package is not available for installation via pip or conda.")
        else:
            st.error(f"ImportError: {e}")
        return None, None
    except HTTPError as e:
        st.error(f"Error downloading model: {e}")
        if '401' in str(e):
            st.error("Authentication failed. Please ensure your access token is correct and has the necessary permissions.")
        return None, None
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        return None, None

# Function to load the model
def load_model(model_path):
    start_load_time = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        end_load_time = time.time()
        load_time = end_load_time - start_load_time
        st.session_state.load_time = load_time
        st.session_state.model_path = model_path
        return tokenizer, model.to('xpu')
    except ImportError as e:
        if 'triton' in str(e):
            st.error("This model requires the 'triton' package, which could not be found in your environment. This package is not available for installation via pip or conda.")
        else:
            st.error(f"ImportError: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Function to perform inference and capture performance metrics
def perform_inference(model, tokenizer, prompt, max_new_tokens=512):
    generated_text = ""
    num_tokens_generated = 0
    start_time = time.time()

    # Measure embedding time
    start_embedding_time = time.time()
    inputs = tokenizer(prompt, return_tensors="pt")
    end_embedding_time = time.time()
    
    inputs = inputs.to(model.device)

    while True:
        start_generation_time = time.time()
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs.input_ids, max_new_tokens=max_new_tokens, generation_config=generation_config)
        end_generation_time = time.time()

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text += response
        num_tokens_generated += len(response.split())

        # Update the input with the generated response for further continuation
        prompt += response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Break if the generated output is less than max_new_tokens, indicating the end of generation
        if len(response.split()) < max_new_tokens:
            break

    end_time = time.time()

    # Performance metrics
    num_tokens = len(tokenizer.encode(prompt))
    duration = end_time - start_time
    tokens_per_sec = num_tokens_generated / duration
    first_token_time = duration / num_tokens_generated if num_tokens_generated > 0 else float('inf')

    cpu_embedding_time = end_embedding_time - start_embedding_time
    gpu_generation_time = end_generation_time - start_generation_time

    return generated_text, tokens_per_sec, num_tokens, num_tokens_generated, duration, first_token_time, cpu_embedding_time, gpu_generation_time

# Sidebar layout for model selection and loading
with st.sidebar:
    st.header("Model Configuration")
    
    model_name = st.selectbox("Select a model:", list(default_models.keys()) + ["Custom Model"])
    if model_name == "Custom Model":
        model_id = st.text_input("Enter custom model ID (from Hugging Face):")
        hf_token = st.text_input("Enter your Hugging Face access token (if needed):", type="password")
    else:
        model_id = default_models[model_name]
        hf_token = ""

    model_path = os.path.join(CACHE_DIR, model_name.replace("/", "_"))
    
    if os.path.exists(model_path):
        st.write(f"Model available locally at: {model_path}")
        # Button to load the local model
        if st.button('Load Model'):
            with st.spinner('Loading Tokenizer and optimizing Model...'):
                tokenizer, model = load_model(model_path)
                if model and tokenizer:
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.success('Successfully loaded Tokenizer and optimized Model!')
                    st.write(f"Model loaded from: {model_path}")
                else:
                    st.error('Failed to load the model.')
    else:
        st.write("Model not found locally.")
        # Button to download and load the model
        if st.button('Download and Load Model'):
            with st.spinner('Downloading and loading the model...'):
                tokenizer, model = download_model(model_id, hf_token)
                if model and tokenizer:
                    model_dir = os.path.join(CACHE_DIR, model_name.replace("/", "_"))
                    model.save_pretrained(model_dir)
                    tokenizer.save_pretrained(model_dir)
                    st.session_state.model = model.to('xpu')
                    st.session_state.tokenizer = tokenizer
                    st.success('Successfully downloaded and loaded the model!')
                    st.write(f"Model downloaded to: {model_dir}")
                    # Verify the files
                    if os.path.exists(model_dir):
                        st.write(f"Model files are located at: {model_dir}")
                        st.write("Files in the directory:")
                        st.write(os.listdir(model_dir))
                    else:
                        st.error(f"Failed to save the model to: {model_dir}")
                else:
                    st.error('Failed to download and load the model.')

# Main layout for chatting with the model
if 'model' in st.session_state and 'tokenizer' in st.session_state:
    st.write("Model is loaded and ready to chat!")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Accept user input
    if prompt := st.chat_input('Ask me anything'):
        # Add user message to chat history
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        # Display user message
        with st.chat_message('user'):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message('assistant'):
            try:
                formatted_prompt = f"user: {prompt}\n\nassistant:"
                
                if st.session_state.get("first_time_load", False):
                    st.session_state.first_time_load = False
                    st.info('Note: The first time that each model runs on Intel iGPU/Intel Arc™ A300-Series or Pro A60, it may take several minutes for GPU kernels to compile and initialize. Please be patient until it finishes warm-up.')
                    
                    # Warm-up
                    _ = perform_inference(st.session_state.model, st.session_state.tokenizer, formatted_prompt, max_new_tokens=32)

                # Actual generation with performance metrics
                response, tokens_per_sec, num_tokens, num_tokens_generated, duration, first_token_time, cpu_embedding_time, gpu_generation_time = perform_inference(st.session_state.model, st.session_state.tokenizer, formatted_prompt)
                
                st.markdown(response)
                st.session_state.messages.append({'role': 'assistant', 'content': response})
                
                # Store metrics in session state
                st.session_state.performance_metrics = {
                    'load_time': st.session_state.get('load_time', 0),
                    'tokens_per_sec': tokens_per_sec,
                    'num_tokens': num_tokens,
                    'num_tokens_generated': num_tokens_generated,
                    'duration': duration,
                    'first_token_time': first_token_time,
                    'cpu_embedding_time': cpu_embedding_time,
                    'gpu_generation_time': gpu_generation_time
                }

            except Exception as e:
                st.error(f"Error during generation: {e}")

# Sidebar button for displaying performance metrics
with st.sidebar:
    if 'performance_metrics' in st.session_state:
        if st.button('Show Performance Metrics'):
            metrics = st.session_state.performance_metrics
            st.write(f"Model Load Time: {metrics['load_time']:.2f} seconds")
            st.write(f"Tokens per second: {metrics['tokens_per_sec']:.2f}")
            st.write(f"Number of tokens in prompt: {metrics['num_tokens']}")
            st.write(f"Number of tokens generated: {metrics['num_tokens_generated']}")
            st.write(f"Duration: {metrics['duration']:.2f} seconds")
            st.write(f"Time per token: {metrics['first_token_time']:.2f} seconds")
            st.write(f"CPU Embedding Time: {metrics['cpu_embedding_time']:.2f} seconds")
            st.write(f"GPU Generation Time: {metrics['gpu_generation_time']:.2f} seconds")
            st.write(f"Model Path: {st.session_state.get('model_path', 'Unknown')}")
    else:
        st.write("Please load a model to start chatting.")
