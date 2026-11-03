from fastapi import APIRouter
from app.api.v1.search_endpoints import router as search_router

api_router = APIRouter()
api_router.include_router(search_router, prefix="/engine", tags=["ERA Engine"])