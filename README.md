# LLaMA-2 chatbot on CPU

## :monocle_face: Description
- This project is a Streamlit chatbot with Langchain deploying a **LLaMA2-7b-chat** model on **Intel® 4th Generation Xeon Scalable Processor CPU**.
- The chatbot has a memory that **remembers every part of the speech**, and allows users to optimize the model using  **Intel® Extension for PyTorch (IPEX) in bfloat16 with graph mode** or **smooth quantization** (A new quantization technique specifically designed for LLMs: [ArXiv link](https://arxiv.org/pdf/2211.10438.pdf)), and expect **up to 2.28x speed-up** compared to stock PyTorch.

- **Note:** The CPU needs to support bfloat16 ops in order to be able to use such optimization. On top of software optimizations, I also introduced some hardware optimizations like non-uniform memory access (NUMA). User needs to **ask for access to LLaMA2** models by following this [link](https://huggingface.co/meta-llama#:~:text=Welcome%20to%20the%20official%20Hugging,processed%20within%201%2D2%20days). When getting approval from Meta, you can generate an authentification token from your HuggingFace account, and use it to load the model.

## :scroll: Getting started

1. Start by cloning the repository:  
```bash
git clone https://github.com/aahouzi/llama2-chatbot-cpu.git
cd llama2-chatbot-cpu
```
2. Create a Python 3.9 conda environment:
```bash
conda create -y -n llama2-chat python=3.9
```
3. Activate the environment:  
```bash
conda activate llama2-chat
```
4. Install requirements for NUMA:  
```bash
conda install -y gperftools -c conda-forge
conda install -y intel-openmp
sudo apt install numactl
```
5. Install the app requirements:  
```bash
pip install -r requirements.txt
```

## :rocket: Start the app

- Default mode (no optimizations):
```bash
bash launcher.sh --script=app/app.py --port=<port> --physical_cores=<physical_cores> --auth_token=<auth_token>
```

- IPEX in graph mode with FP32:
```bash
bash launcher.sh --script=app/app.py --port=<port> --physical_cores=<physical_cores> --auth_token=<auth_token> --ipex --jit
```

- IPEX in graph mode with bfloat16:
```bash
bash launcher.sh --script=app/app.py --port=<port> --physical_cores=<physical_cores> --auth_token=<auth_token> --dtype=bfloat16 --ipex --jit
```

- Smooth quantization:
```bash
bash launcher.sh --script=app/app.py --port=<port> --physical_cores=<physical_cores> --auth_token=<auth_token> --sq
```

## :computer: Chatbot demo

<video src='https://github.com/aahouzi/llama2-chatbot-cpu/assets/112881240/c455fe0e-224f-4182-8d58-7d6985e922af' width=180/>

## :mailbox_closed: Contact
For any information, feedback or questions, please [contact me][anas-email]









[anas-email]: mailto:ahouzi2000@hotmail.fr


