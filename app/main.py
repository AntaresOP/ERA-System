from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.router import api_router
from app.api.v1.search_endpoints import hybrid_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # App Startup: sync BM25 index on launch
    print("Initializing ERA System Engine Services...")
    try:
        hybrid_pipeline.sync_sparse_index()
        print("Sparse index synchronized successfully.")
    except Exception as e:
        print(f"Index sync warning during startup: {e}")
    yield
    # App Shutdown logic
    print("Shutting down ERA System Engine Services.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Enable CORS for frontend client connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Health Check"])
async def root_health_check():
    return {
        "status": "online",
        "system": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "version": "1.0.0"
    }