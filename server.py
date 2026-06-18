"""
Zentrix GPU Server v10 — Async job handling
POST /generate → returns job_id immediately
GET /status/{job_id} → returns status + result when done
"""
import os, time, traceback, torch, uuid, threading
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from handler import EndpointHandler

print("=" * 50)
print("🚀 Zentrix GPU Server v10 (async)")
print(f"  torch=={torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
print("=" * 50)

# In-memory job storage
jobs = {}
handler = None

@asynccontextmanager
async def lifespan(app):
    global handler
    handler = EndpointHandler()
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/health")
@app.get("/")
def health():
    active = sum(1 for j in jobs.values() if j["status"] in ("queued", "running"))
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "version": "v10-async",
        "active_jobs": active,
    }


def _run_job(job_id: str, data: dict):
    """Runs handler in background thread."""
    jobs[job_id]["status"] = "running"
    t0 = time.time()
    try:
        result = handler(data)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result
        print(f"✅ Job {job_id} done in {time.time()-t0:.0f}s")
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["result"] = {"error": str(e), "traceback": traceback.format_exc()}
        print(f"❌ Job {job_id} error: {e}")


@app.post("/generate")
async def generate_async(request: Request):
    """Accept job, return job_id immediately. Worker polls /status/{job_id}."""
    if handler is None:
        return JSONResponse(status_code=503, content={"error": "Not ready"})
    try:
        data = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "queued", "result": None, "created": time.time()}

    t = threading.Thread(target=_run_job, args=(job_id, data), daemon=True)
    t.start()

    model = data.get("inputs", {}).get("model", "?")
    print(f"📥 Job {job_id} queued (model={model})")
    return JSONResponse(content={"job_id": job_id, "status": "queued"})


@app.get("/status/{job_id}")
def get_status(job_id: str):
    """Poll endpoint — returns status and result when done."""
    job = jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    resp = {"job_id": job_id, "status": job["status"]}

    if job["status"] in ("done", "error"):
        resp["result"] = job["result"]
        # Auto-cleanup completed jobs older than 10 minutes
        if time.time() - job["created"] > 600:
            jobs.pop(job_id, None)

    return JSONResponse(content=resp)


# Cleanup old jobs periodically
def _cleanup_old_jobs():
    while True:
        time.sleep(300)
        now = time.time()
        stale = [k for k, v in jobs.items() if now - v["created"] > 1800]
        for k in stale:
            jobs.pop(k, None)
        if stale:
            print(f"🧹 Cleaned {len(stale)} old jobs")

threading.Thread(target=_cleanup_old_jobs, daemon=True).start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🌐 Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=600)
