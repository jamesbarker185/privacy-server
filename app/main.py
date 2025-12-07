from fastapi import FastAPI
from app.api.routes import router
from app.core.config import settings
import structlog

app = FastAPI(title=settings.PROJECT_NAME)

# Setup logging
structlog.configure(
    processors=[
        structlog.processors.JSONRenderer()
    ]
)

app.include_router(router, prefix=settings.API_V1_STR)

@app.get("/health")
def health_check():
    return {"status": "ok"}
