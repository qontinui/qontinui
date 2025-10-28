"""Health check and version endpoints.

This router provides system health monitoring and version information endpoints.

NO backward compatibility - clean FastAPI code.
"""

import sys

from fastapi import APIRouter
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


class VersionResponse(BaseModel):
    """Version information response."""

    version: str
    api_version: str
    python_version: str


router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.

    Returns:
        HealthResponse with status and version
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
    }


@router.get("/version", response_model=VersionResponse)
async def get_version():
    """Get version information.

    Returns:
        VersionResponse with version details
    """
    return {
        "version": "1.0.0",
        "api_version": "1.0",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
