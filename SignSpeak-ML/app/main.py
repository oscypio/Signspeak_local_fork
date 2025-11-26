from fastapi import FastAPI
from .api import router as api_router

app = FastAPI(
    title="ASL Recognition API",
    version="1.0.0"
)

# Healthcheck
@app.get("/health")
def healthcheck():
    return {"status": "ok"}

# API
app.include_router(api_router, prefix="/api")
