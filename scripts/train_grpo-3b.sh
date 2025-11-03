#!/bin/bash
set -x

GPU_NUM=8
USE_EXPERIENCE=true # Set to false to disable experience database

export DATA_DIR='data/nq_hotpotqa_train'

WAND_PROJECT='EvolveR'
export SWANLAB_LOG_DIR="swanlog"

export BASE_MODEL='models/evolver/qwen-3b-instruct-cold_start'
export EXPERIMENT_NAME=nq_hotpotqa-evolver-3b

export EMBEDDING_API_URL="http://127.0.0.1:8081/v1"
export RETRIEVE_URL="http://127.0.0.1:8000/retrieve"

export EXPERIENCE_EXPORT_DIR="data/evolver/result"
 



# --- MilvusDB Service Management ---
if [ "$USE_EXPERIENCE" = "true" ]; then
    DB_SERVER_DIR="${EXPERIENCE_EXPORT_DIR}/${EXPERIMENT_NAME}/db_server"
    DB_SERVER_LOG_FILE="${DB_SERVER_DIR}/db_server-${EXPERIMENT_NAME}.log"
    DB_EXPORT_DIR="${EXPERIENCE_EXPORT_DIR}/${EXPERIMENT_NAME}/db_exports"


    export VDB_SERVER_URL="http://127.0.0.1:8080"

    cleanup_db_server() {
        echo "--- Cleaning up MilvusDB Server ---"
        
        # Export data before shutting down
        echo "Exporting database collections to ${DB_EXPORT_DIR}..."
        curl -s -X POST "${VDB_SERVER_URL}/export/" \
          -H "Content-Type: application/json" \
          -d "{
            \"collections\": [\"principles\", \"trajectories\"],
            \"format\": \"jsonl\",
            \"output_root_dir\": \"${EXPERIENCE_EXPORT_DIR}\",
            \"experiment_name\": \"${EXPERIMENT_NAME}\"
          }" || true
        echo -e "\nDatabase export command sent."
        echo "--- Cleanup complete ---"
    }

    # run the cleanup function on script exit or interruption
    trap cleanup_db_server EXIT SIGINT SIGTERM

    echo "--- Wiping old MilvusDB data before start ---"
    rm -rf $DB_SERVER_DIR

    mkdir -p $DB_SERVER_DIR
    mkdir -p $DB_EXPORT_DIR

    echo "--- Starting MilvusDB Server for experiment: ${EXPERIMENT_NAME} ---"

    export VDB_BASE_DIR="$DB_SERVER_DIR"
    export VDB_IMPORT_DB_FILE="${VDB_IMPORT_DB_FILE}"
    bash evolver/experience/milvusdb/start_server.sh > "$DB_SERVER_LOG_FILE" 2>&1 &

    echo "Waiting for DB server (${VDB_SERVER_URL}) to start..."
    for i in $(seq 1 60); do
      if curl -s "${VDB_SERVER_URL}/" | grep -q '"status":"running"'; then
        echo "DB Server is running."
        break
      fi
      sleep 2
      if [ $i -eq 60 ]; then
        echo "Error: DB server failed to start within timeout. Check log file: $DB_SERVER_LOG_FILE"
        exit 1
      fi
    done
else
    echo "--- Experience DB is disabled. Skipping MilvusDB server setup. ---"
    export VDB_SERVER_URL="" 
fi


export VLLM_ATTENTION_BACKEND=XFORMERS
export MKL_SERVICE_FORCE_INTEL=1
export HYDRA_FULL_ERROR=1


python3 -m verl.trainer.main_ppo \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  data.train_data_num=null \
  data.val_data_num=null \
  data.train_batch_size=128 \
  data.val_batch_size=1024 \
  data.max_prompt_length=8192 \
  data.max_response_length=1024 \
  data.max_start_length=2048 \
  data.max_obs_length=2048 \
  data.shuffle_train_dataloader=true \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path=$BASE_MODEL \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size=32 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  algorithm.no_think_rl=false \
  algorithm.state_masking.mask_sections=['information','experience'] \
  actor_rollout_ref.rollout.n_agent=8 \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.actor.state_masking=true \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  +trainer.val_only=false \
  +trainer.val_before_train=false \
  trainer.val_do_sample=false \
  trainer.val_temperature=0.6 \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=${GPU_NUM} \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.project_name=$WAND_PROJECT \
  trainer.experiment_name=$EXPERIMENT_NAME \
  trainer.total_epochs=5 \
  trainer.total_training_steps=1000 \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir=models/save_models/evolver/$EXPERIMENT_NAME \
  rewards.weights.format=0.1 \
  rewards.weights.outcome=1.0 \
  rewards.weights.info_gain=0 \
  rewards.weights.experience=0 \
  experience.enable=$USE_EXPERIENCE \
  experience.vdb_server_url=$VDB_SERVER_URL \
  experience.organize_interval=1 \
  experience.export_interval=50 \
  experience.clean_low_metric_threshold=0.3 \
  experience.clean_interval=10 \
  experience.experience_data_dir=${EXPERIENCE_EXPORT_DIR} \
  experience.embedding_api_url=${EMBEDDING_API_URL} \
  experience.trajectory_choice_ratio=0.25 \
  experience.retrieve_component.principle=true \
  experience.retrieve_component.structure=true \
  experience.retrieve_component.success_trajectory=false \
  experience.retrieve_component.failure_trajectory=false \
  max_turns=10 \
  retriever.url=${RETRIEVE_URL} \
  retriever.topk=3 \
  2>&1 | tee $EXPERIENCE_EXPORT_DIR/$EXPERIMENT_NAME/train_log.log