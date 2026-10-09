"""Portable OpenCode -> OpenAI-compatible Muse Spark gateway."""
from .gateway import OpenCodeClient, create_server
__all__ = ["OpenCodeClient", "create_server"]
