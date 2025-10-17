GPU_NUM=2

vllm serve models/bge-m3 --served-model-name bge_m3 --port 8081 --tensor-parallel-size ${GPU_NUM} --max-model-len 8192 --uvicorn-log-level debug --disable-log-requests

