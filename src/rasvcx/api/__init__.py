"""RASVC-X HTTP API package (Module 14).

Public surface:
  create_app(settings) -- construct the FastAPI application
  app                  -- module-level instance for uvicorn

Usage:
  uvicorn rasvcx.api.main:app --host 0.0.0.0 --port 8000
"""

from rasvcx.api.main import app, create_app

__all__ = ["create_app", "app"]