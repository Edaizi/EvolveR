#!/bin/bash
set -x

export RAY_ADDRESS=''


GPU_NUM=8
USE_EXPERIENCE=true # Set to false to disable experience DB and related features

export DATA_DIR='data/exp-rl/nq_hotpotqa_train'

WAND_PROJECT='EvolveR'
export SWANLAB_LOG_DIR="swanlog"


# To restore a DB from a previous experiment's .db file, set this path.
# Example: export VDB_IMPORT_DB_FILE="/path/to/previous/experiment/db_server/milvus_exp.db"
export VDB_IMPORT_DB_FILE="" # Leave empty for normal runs


export SWANLAB_LOG_DIR="swanlog"


export TEST_FILE='test.parquet'


export BASE_MODEL=''


# export VDB_IMPORT_PRINCIPLES=''
# export VDB_IMPORT_TRAJECTORIES=''

export EXPERIMENT_NAME=




export PARTITION=""
export EMBEDDING_API_URL=""


export RETRIEVE_URL=""

export EXPERIENCE_EXPORT_DIR="data/evolver/result"
 
export HYDRA_FULL_ERROR=1


# --- MilvusDB Service Management ---
if [ "$USE_EXPERIENCE" = "true" ]; then
    DB_SERVER_DIR="${EXPERIENCE_EXPORT_DIR}/${EXPERIMENT_NAME}/db_server"
    DB_SERVER_LOG_FILE="${DB_SERVER_DIR}/db_server-${EXPERIMENT_NAME}.log"
    DB_EXPORT_DIR="${EXPERIENCE_EXPORT_DIR}/${EXPERIMENT_NAME}/db_exports"

    # Pick a compute node for DB server and resolve its IP (managed by Slurm)
    # This ensures DB server does not run on the management node.
    EXCLUDE_FLAG=""
    if [ -n "$EXCLUDE_DB_NODES" ]; then
        EXCLUDE_FLAG="--exclude=${EXCLUDE_DB_NODES}"
    fi
    DB_NODE=$(srun --nodes=1 --ntasks=1 -p $PARTITION $EXCLUDE_FLAG hostname | tail -n1)
    DB_NODE_IP=$(srun --nodes=1 --ntasks=1 -p $PARTITION -w "$DB_NODE" hostname -i | awk '{print $1}')
    DB_SERVER_URL="http://${DB_NODE_IP}:8080"
    # Export for python processes (driver + ray workers)
    export VDB_SERVER_URL="${DB_SERVER_URL}"


    echo "Planned DB node: ${DB_NODE} (${DB_NODE_IP})"

    # Function to clean up the database server
    cleanup_db_server() {
        echo "--- Cleaning up MilvusDB Server ---"
        
        # Export data before shutting down
        echo "Exporting database collections to ${DB_EXPORT_DIR}..."
        curl -s -X POST "${DB_SERVER_URL}/export/" \
          -H "Content-Type: application/json" \
          -d "{
            \"collections\": [\"principles\", \"trajectories\"],
            \"format\": \"jsonl\",
            \"output_root_dir\": \"${EXPERIENCE_EXPORT_DIR}\",
            \"experiment_name\": \"${EXPERIMENT_NAME}\"
          }" || true
        echo -e "\nDatabase export command sent."

        DB_SERVER_JOBID=$(awk '/phoenix-srun: Job [0-9]+ scheduled successfully!/ {for(i=1;i<=NF;i++){if($i=="Job"){print $(i+1)}}}' "$DB_SERVER_LOG_FILE" | head -n1)
        if [ ! -z "$DB_SERVER_JOBID" ]; then
            echo "Killing DB server launcher (srun) with JOBID: $DB_SERVER_JOBID"
            scancel $DB_SERVER_JOBID
        else
            echo "Warning: Could not find DB server JOBID in log file, fallback to killing local PID: $DB_SERVER_PID"
            if [ ! -z "$DB_SERVER_PID" ]; then
                kill $DB_SERVER_PID 2>/dev/null || true
                wait $DB_SERVER_PID 2>/dev/null || true
            fi
        fi
        echo "--- Cleanup complete ---"
    }

    # Set a trap to run the cleanup function on script exit or interruption
    trap cleanup_db_server EXIT SIGINT SIGTERM

    export VDB_AUTO_IMPORT=1

    mkdir -p $DB_SERVER_DIR
    mkdir -p $DB_EXPORT_DIR

    echo "--- Starting MilvusDB Server for experiment: ${EXPERIMENT_NAME} ---"
    # Start the DB server on the selected compute node via Slurm
    srun --nodes=1 --ntasks=1 -p $PARTITION -w "$DB_NODE" \
      --export=ALL,VDB_BASE_DIR="$DB_SERVER_DIR",VDB_IMPORT_DB_FILE="$VDB_IMPORT_DB_FILE",VDB_IMPORT_PRINCIPLES="$VDB_IMPORT_PRINCIPLES",VDB_IMPORT_TRAJECTORIES="$VDB_IMPORT_TRAJECTORIES" \
      bash evolver/experience/milvusdb/start_server.sh > "$DB_SERVER_LOG_FILE" 2>&1 &
    DB_SERVER_PID=$!
    echo "DB Server launcher started (srun) with PID: $DB_SERVER_PID. Log file: $DB_SERVER_LOG_FILE"

    # Robust wait for the server to become ready (up to ~120s)
    echo "Waiting for DB server (${DB_SERVER_URL}) to start..."
    for i in $(seq 1 60); do
      if curl -s "${DB_SERVER_URL}/" | grep -q '"status":"running"'; then
        echo "DB Server is running."
        break
      fi
      sleep 2
      if [ $i -eq 60 ]; then
        echo "Error: DB server failed to start within timeout. Check log file: $DB_SERVER_LOG_FILE"
        exit 1
      fi
    done
    # --- End of MilvusDB Service Management ---
else
    echo "--- Experience DB is disabled. Skipping MilvusDB server setup. ---"
    export VDB_SERVER_URL="" 
fi


# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

ray job submit --runtime-env-json '{"excludes": ["logs", "ray_log", "swanlog"]}' -- \
    python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/$TEST_FILE \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=128 \
    data.val_batch_size=2048 \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    data.max_start_length=2048 \
    data.max_obs_length=2048 \
    data.shuffle_train_dataloader=true \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-8 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab','wandb'] \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.val_do_sample=false \
    trainer.val_temperature=0.6 \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=${GPU_NUM} \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=15 \
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
    2>&1 | tee $EXPERIENCE_EXPORT_DIR/$EXPERIMENT_NAME/test_logs.log

