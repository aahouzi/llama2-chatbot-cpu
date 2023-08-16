#!/bin/bash
set -x

FILE=./models/best_model.pt
OUTPUT_DIR=./models
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
  prompt="Once upon a time, there was"
  num_warmup=15
  alpha=0.8
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

    # Activate env
    source ~/anaconda3/bin/activate amall

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
    
    # Check if there is a quantized model, when user selects smooth quantization
    if [ $extra_cmd =~ "--sq" ] && ! [ -f "$FILE" ]; then
        echo -e '\n[INFO]: Quantized model not detected, launching SmoothQuant process..\n'
        python3 smoothquant/run_generation.py --model ${model_id} --output_dir ${OUTPUT_DIR} --alpha ${alpha} --auth_token ${auth_token} --quantize --sq --ipex
        echo -e '\n[INFO]: Starting SmoothQuant performance evaluation..\n'
        python3 smoothquant/run_generation.py --model ${model_id} --output_dir ${OUTPUT_DIR} --auth_token ${auth_token} --benchmark --ipex --int8
        echo -e '\n[INFO]: Starting SmoothQuant accuracy evaluation..\n'
        python3 smoothquant/run_generation.py --model ${model_id} --output_dir ${OUTPUT_DIR} --tasks lambada_openai --batch_size 112 --accuracy --int8 --ipex
    fi

    echo -e '\n[INFO]: Starting streamlit app..\n'
    echo -e ${prompt}
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
