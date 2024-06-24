Intel GPU (xpu) devices Based Local LLM chatbot

This project shows the steps to enable your Intel GPU (recent Xe/XPU - iGPU or dGPU devices) to run the local chat bot and Local RAG systems.
Most of the steps are referred from https://ipex-llm.readthedocs.io/en/latest/index.html,
This website provides the complete steps about enabling Intel GPU using Intel PyTorch Extention.

When you try to load the model to Intel GPU devices, it requires to recompile the LLM kernel for its XPU devices using CL/Computing resources. So it may take a while but after first time, it gives faster response.

My program allow the user to select the model (custom or huggingface) and download or local and load them into Intel GPU (based on iGPU or dGPU). After the model load is completed, user can start chat with the model.
This will work with latest gen GPUs and it requires atleast 31.0.101.5122 or higher driver version.

I created a v=batch script and it will check and let the user know the version and update requirement. User can download from https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html
The batch script will allow user to download, install conda miniforge package and create a virtual environment, install all the required pip packages and run the streamlit application.

The script will install miniforge, and create a new python environment llm:

conda create -n llm python=3.11 libuv
Activate the newly created environment llm:
conda activate llm

It will download the packages automatically
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

It will also set the env variables based on GPU type. XMX is supported in Intel Arc Products

Set the following environment variables according to your device:

For Intel iGPU:
set SYCL_CACHE_PERSISTENT=1
set BIGDL_LLM_XMX_DISABLED=1

For Intel Arc™ A770:
set SYCL_CACHE_PERSISTENT=1

The script will run and bring the UI for user to select the model and if required, user needs to enter their access token for huggingface to download the model. Once download and load the model to xpu, user can chat with the selected model.

