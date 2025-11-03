import logging
import uvicorn
from fastapi import FastAPI
import os
from datetime import datetime
from models import StatusResponse
from dependencies import initialize_clients, shutdown_clients, get_principle_client, get_trajectory_client
from routers import initial, principles, trajectories, search, export_import
from config import BASE_DIR, LOG_FILE

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Vector Database Service",
    description="FastAPI service for managing experience principles and trajectories with vector search",
    version="1.0.0"
)


def _log_collection_counts(principle_client, trajectory_client, context: str):
    def _count(client, cname):
        try:
            stats = client.client.get_collection_stats(cname)
            # stats may be dict or json-like string depending on client version
            if isinstance(stats, dict):
                rc = stats.get('row_count') or stats.get('rowCount')
            else:
                # try to parse when it's a string like '{"row_count": 123}'
                import json as _json
                rc = _json.loads(stats).get('row_count')
            return int(rc) if rc is not None else None
        except Exception:
            try:
                # fallback (approx) — will undercount if server-side default limit < total
                res = client.client.query(collection_name=cname, filter="", limit=5)
                return f">= {len(res)}"
            except Exception:
                return "unknown"

    p_name = principle_client.collection_name
    t_name = trajectory_client.collection_name
    p_cnt = _count(principle_client, p_name)
    t_cnt = _count(trajectory_client, t_name)
    logger.info(f"[{context}] Collection counts: principles({p_name})={p_cnt}, trajectories({t_name})={t_cnt}")


@app.on_event("startup")
async def startup_event():
    from pathlib import Path
    experiment_name = os.environ.get("EXPERIMENT_NAME")
    initialize_clients(experiment_name, embedding_api_url=os.environ.get("EMBEDDING_API_URL"),
                        embedding_api_key=os.environ.get("EMBEDDING_API_KEY"),
                        embedding_model=os.environ.get("EMBEDDING_MODEL_NAME", "bge_m3"))

    try:
        auto_import = os.environ.get("VDB_AUTO_IMPORT", "0") == "1"
        import_format = os.environ.get("VDB_IMPORT_FORMAT", "jsonl")
        p_file = os.environ.get("VDB_IMPORT_PRINCIPLES")
        t_file = os.environ.get("VDB_IMPORT_TRAJECTORIES")

        if auto_import or p_file or t_file:
            logger.info("Auto-import enabled on startup")
            p_client = get_principle_client()
            t_client = get_trajectory_client()
            if p_file:
                await export_import.import_data(p_file, "principles", import_format, p_client, t_client)
            if t_file:
                await export_import.import_data(t_file, "trajectories", import_format, p_client, t_client)
            if not p_file and not t_file and experiment_name:
                sanitized = experiment_name.replace('-', '_')
                base_dir = Path("/mnt/petrelfs/wurong/workspace/evolver/data/evolver/result") / sanitized / "db_exports"
                p_candidates = list(base_dir.glob("principles_*.jsonl"))
                t_candidates = list(base_dir.glob("trajectories_*.jsonl"))
                if p_candidates:
                    await export_import.import_data(str(p_candidates[-1]), "principles", "jsonl", p_client, t_client)
                if t_candidates:
                    await export_import.import_data(str(t_candidates[-1]), "trajectories", "jsonl", p_client, t_client)
            # Log post-import counts
            _log_collection_counts(p_client, t_client, context="auto-import")
    except Exception as e:
        logger.warning(f"Auto-import on startup failed or skipped: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    shutdown_clients()


@app.get("/", response_model=StatusResponse)
async def root():
    return StatusResponse(
        status="running",
        message="Vector Database Service is running",
        timestamp=datetime.now().isoformat()
    )


app.include_router(initial.router)
app.include_router(principles.router)
app.include_router(trajectories.router)
app.include_router(search.router)
app.include_router(export_import.router)

if __name__ == "__main__":
    logger.info(f"Starting Vector Database Service...")
    logger.info(f"Database directory: {BASE_DIR}")
    logger.info(f"Log file: {LOG_FILE}")
    
    uvicorn.run(
        "db_server:app",
        host="0.0.0.0",
        port=8007,  # 8080
        reload=False,
        log_level="info"
    )
