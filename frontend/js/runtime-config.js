// Runtime-injected frontend configuration.
// This file is overwritten in Render static-site builds by scripts/render_frontend_build.py.
window.ELEVATE_RUNTIME_CONFIG = window.ELEVATE_RUNTIME_CONFIG || {};
window.__ELEVATE_API_BASE_URL__ = window.ELEVATE_RUNTIME_CONFIG.API_BASE_URL || window.__ELEVATE_API_BASE_URL__ || '';
