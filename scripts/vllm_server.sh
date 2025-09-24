HOSTNAME=$(hostname)
IP=$(hostname -I | awk '{print $1}')

echo "服务将在节点 $HOSTNAME ($IP) 上启动"

GPU_NUM=2

# vllm serve models/Qwen/Qwen2.5-0.5B-Instruct --tensor-parallel-size 2 --served-model-name Qwen2.5-0.5b --host 0.0.0.0 --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.7 --uvicorn-log-level debug --disable-log-requests
# vllm serve /mnt/phwfile/datafrontier/wangxiaoman/coldstart/global_step_49 --tensor-parallel-size 2 --served-model-name Qwen2.5-0.5b --host 0.0.0.0 --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.7 --uvicorn-log-level debug --disable-log-requests


# ## 在vllm1环境启动
# vllm serve /mnt/phwfile/datafrontier/zhangyaoze/vllm_weight/bge-m3 --served-model-name bge_m3 --port 8081 --tensor-parallel-size ${GPU_NUM} --max-model-len 8192 --uvicorn-log-level debug --disable-log-requests


vllm serve /mnt/phwfile/datafrontier/wurong/models/Qwen/Qwen3-32B --tensor-parallel-size 8 --served-model-name Qwen3-32b --host 0.0.0.0 --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.6 --uvicorn-log-level debug --disable-log-requests