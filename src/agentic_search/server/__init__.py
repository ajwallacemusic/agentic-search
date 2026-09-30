"""HTTP service for agentic search (install the `server` extra: fastapi, uvicorn)."""

from agentic_search.server.app import create_app
from agentic_search.server.config import (
    AuthConfig,
    Profile,
    ProfileLimits,
    ServiceConfig,
    check_profiles,
    load_service,
)

__all__ = ["AuthConfig", "Profile", "ProfileLimits", "ServiceConfig", "check_profiles",
           "create_app", "load_service"]
