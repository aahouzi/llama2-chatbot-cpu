#!/bin/bash
set -x

CONDA_ENV=amall
FILE=./models/best_model.pt
PHYSICAL_CORES=56

function main {

  init_params "$@"
  start_app

}

# init params
function init_params {
  script="app/app.py"
  port="9000"
  auth_token=""
  model_id="meta-llama/Llama-2-7b-chat-hf"
  window_len=5
  dtype="float32"
  device="cpu"
  max_new_tokens=32
  prompt="Once upon time, there was"
  num_warmup=15
  alpha="auto"
  extra_cmd=""
  for var in "$@"
  do
    case $var in
      --script=*)
          script=$(echo $var | cut -f2 -d=)
      ;;
      --port=*)
          port=$(echo $var | cut -f2 -d=)
      ;;
      --auth_token=*)
          auth_token=$(echo $var | cut -f2 -d=)
      ;;
      --model_id=*)
          model_id=$(echo $var | cut -f2 -d=)
      ;;
      --window_len=*)
          window_len=$(echo $var | cut -f2 -d=)
      ;;
      --dtype=*)
          dtype=$(echo $var | cut -f2 -d=)
      ;;
      --device=*)
          device=$(echo $var | cut -f2 -d=)
      ;;
      --max_new_tokens=*)
          max_new_tokens=$(echo $var | cut -f2 -d=)
      ;;
      --prompt=*)
          prompt=$(echo $var | cut -f2 -d=)
      ;;
      --num_warmup=*)
          num_warmup=$(echo $var | cut -f2 -d=)
      ;;
      --alpha=*)
          alpha=$(echo $var | cut -f2 -d=)
      ;;
      --ipex)
          extra_cmd=$extra_cmd" --ipex"
      ;;
      --jit)
          extra_cmd=$extra_cmd" --jit"
      ;;
      --sq)
          extra_cmd=$extra_cmd" --sq"
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}


function start_app {

    if [[ $extra_cmd =~ "--ipex" ]]; then
        echo -e '\n[INFO]: Installing IPEX llm branch..\n'
        python3 -m pip install torch==2.1.0.dev20230711+cpu torchvision==0.16.0.dev20230711+cpu torchaudio==2.1.0.dev20230711+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
        python3 -m pip install https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_dev/cpu/intel_extension_for_pytorch-2.1.0.dev0%2Bcpu.llm-cp39-cp39-linux_x86_64.whl
        conda install -y libstdcxx-ng=12 -c conda-forge
    fi
        
    # Check if there is a quantized model, when user selects smooth quantization
    if [[ $extra_cmd =~ "--sq" ]]; then
        echo -e '\n[INFO]: Re-installing app requirements to ensure PyTorch version is compatible..\n'
        pip install requirements.txt
        if ! [ -f "$FILE" ]; then
            echo -e '\n[INFO]: Quantized model not detected, launching SmoothQuant process..\n'
            conda create -y -n smoothquant python=3.9 
            eval "$(conda shell.bash hook)" 
            conda activate smoothquant 
            pip install -r smoothquant/requirements.txt
            python3 smoothquant/run_generation.py --model ${model_id} --alpha ${alpha} --auth_token ${auth_token} --quantize --sq --ipex
            echo -e '\n[INFO]: Starting SmoothQuant performance evaluation..\n'
            python3 smoothquant/run_generation.py --model ${model_id} --auth_token ${auth_token} --benchmark --ipex --int8
            echo -e '\n[INFO]: Starting SmoothQuant accuracy evaluation..\n'
            # TODO: Fix evaluation issue
            # python3 smoothquant/run_generation.py --model ${model_id} --auth_token ${auth_token} --batch_size 112 --accuracy --int8 --ipex
            conda deactivate && conda env remove -n smoothquant && conda activate $CONDA_ENV
        fi
    fi
    
    # Setup environment variables for performance on Xeon
    export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6
    export KMP_BLOCKTIME=INF
    export KMP_TPAUSE=0
    export KMP_SETTINGS=1
    export KMP_AFFINITY=granularity=fine,compact,1,0
    export KMP_FORJOIN_BARRIER_PATTERN=dist,dist
    export KMP_PLAIN_BARRIER_PATTERN=dist,dist
    export KMP_REDUCTION_BARRIER_PATTERN=dist,dist
    export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so # Intel OpenMP
    export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
    export OMP_NUM_THREADS=${PHYSICAL_CORES}

    echo -e '\n[INFO]: Starting streamlit app..\n'
    numactl -m 0 -C 0-$(($PHYSICAL_CORES-1)) streamlit run ${script} \
                                       --server.port=${port} -- \
                                       --auth_token=${auth_token} \
                                       --model_id=${model_id} \
                                       --window_len=${window_len} \
                                       --dtype=${dtype} \
                                       --device=${device} \
                                       --max_new_tokens=${max_new_tokens} \
                                       --prompt="${prompt}" \
                                       --num_warmup=${num_warmup} \
                                       ${extra_cmd}

}

main "$@"
