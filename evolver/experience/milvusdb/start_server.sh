#!/bin/bash
# Description: This script starts the MilvusDB server inside an Apptainer container.
# It sources conda, activates the environment, and then runs the FastAPI server.
# The EXPERIMENT_NAME environment variable should be set before calling this script
# to ensure the database collections are named correctly for the experiment.

# Execute the server startup sequence within a single non-interactive bash shell
# inside the apptainer container. This makes it suitable for background execution.
apptainer exec --nv --bind "/mnt:/mnt" /mnt/petrelfs/wurong/glibc_ubuntu22.sif /bin/bash -c "
export EXPERIMENT_NAME=${EXPERIMENT_NAME}
# Replace all hyphens with underscores in experiment name to comply with Milvus collection naming rules
export EXPERIMENT_NAME=\${EXPERIMENT_NAME//-/_}
# Pass through VDB paths if they are set from srun
export VDB_BASE_DIR=${VDB_BASE_DIR}
# Auto import controls (optional)
export VDB_AUTO_IMPORT=${VDB_AUTO_IMPORT:-0}
export VDB_IMPORT_FORMAT=${VDB_IMPORT_FORMAT:-jsonl}
export VDB_IMPORT_PRINCIPLES=${VDB_IMPORT_PRINCIPLES:-}
export VDB_IMPORT_TRAJECTORIES=${VDB_IMPORT_TRAJECTORIES:-}
# Embedding API configs (can be overridden by outer env)
export EMBEDDING_API_URL=${EMBEDDING_API_URL:-http://10.140.37.36:8000/v1}
export EMBEDDING_API_KEY=${EMBEDDING_API_KEY:-empty}
export EMBEDDING_MODEL=${EMBEDDING_MODEL:-bge_m3}
export VDB_IMPORT_DB_FILE=${VDB_IMPORT_DB_FILE:-}

source /mnt/petrelfs/wurong/miniconda3/etc/profile.d/conda.sh
conda activate exp-rl
cd /mnt/petrelfs/wurong/workspace/evolver/evolver/experience/milvusdb

# Restore from a .db file if specified
if [ -n "\$VDB_IMPORT_DB_FILE" ] && [ -f "\$VDB_IMPORT_DB_FILE" ]; then
    if [ -z "\$VDB_BASE_DIR" ]; then
        echo Warning: VDB_IMPORT_DB_FILE is set, but VDB_BASE_DIR is not defined. Cannot restore. Skipping.
    else
        DEST_DB_FILE="\$VDB_BASE_DIR/milvus_exp.db"
        echo --- Restoring VDB from .db file ---
        echo Source: \$VDB_IMPORT_DB_FILE
        echo Destination: \$DEST_DB_FILE
        
        mkdir -p "\$VDB_BASE_DIR"
        cp -f "\$VDB_IMPORT_DB_FILE" "\$DEST_DB_FILE"
        
        if [ \$\? -eq 0 ]; then
            echo Successfully restored database file.
        else
            echo Warning: Failed to restore database file. Continuing with existing or new DB.
        fi
        echo -----------------------------------
    fi
fi

echo Starting DB server for experiment: \${EXPERIMENT_NAME}
python db_server.py
"