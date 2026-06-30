"""Elevate backend package initializer.

This file makes `backend` a Python package so tests and imports can use
`from backend import ...` style imports reliably.
"""

__all__ = ["app", "models", "routes", "config"]
