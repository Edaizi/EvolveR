#!/bin/bash

CONDA_ENV_NAME="retriever"

SETUP_SCRIPT_PATH=""
CONTAINER_IMAGE_PATH=""

# export CUDA_VISIBLE_DEVICES=0
echo "Starting retrieval server..."
# echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

HOSTNAME=$(hostname)
IP=$(hostname -I | awk '{print $1}')
RETRIEVER_INFO_FILE="retriever_server_info.txt"


cat > $RETRIEVER_INFO_FILE << EOF

RETRIEVER_HOST=$HOSTNAME
RETRIEVER_IP=$IP
RETRIEVER_PORT=8000
RETRIEVER_URL=http://$IP:8000
EOF


file_path=data/Wiki-corpus-embedd/
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=models/intfloat/e5-base-v2


apptainer exec --nv --bind /mnt:/mnt ${CONTAINER_IMAGE_PATH} \
    bash -c "source ${SETUP_SCRIPT_PATH} ${CONDA_ENV_NAME} && python evolver/search/retrieval_server.py \
                                            --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu"
