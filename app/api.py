"""FastAPI backend for ViKIS search and retrieval endpoints."""

from fastapi import FastAPI

app = FastAPI(title="ViKIS API", version="0.1.0")


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint for API discovery."""
    return {"message": "ViKIS API is running."}
