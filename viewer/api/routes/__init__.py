from __future__ import annotations

from viewer.api.routes.conversation import router as conversation_router
from viewer.api.routes.files import router as files_router
from viewer.api.routes.records import router as records_router
from viewer.api.routes.status import router as status_router

__all__ = [
    "conversation_router",
    "files_router",
    "records_router",
    "status_router",
]
