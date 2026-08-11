#!/usr/bin/env python3
"""
frontend_slug_server.py

Local-dev static file server for the Elevate frontend that mirrors the
slug-fallback routing already implemented in backend/app.py, so that
URLs like:

    http://127.0.0.1:8000/nhitmenv/dashboard.html

resolve to the real file:

    frontend/dashboard.html

This is a drop-in replacement for `python -m http.server`, which has no
routing logic at all and 404s the moment a path segment (the school
slug) isn't a literal folder on disk.

Usage:
    python frontend_slug_server.py --directory frontend --port 8000

Behavior (mirrors app.py exactly):
  - "/"                       -> index.html
  - "/<name>.<ext>"           -> served directly (plain top-level file,
                                 e.g. "/dashboard.html")
  - "/<slug>" or "/<slug>/"   -> 302 redirect to "/index.html"
                                 (no dot -> bare tenant slug, not a file)
  - "/<slug>/index.html"      -> 302 redirect to "/index.html"
  - "/<slug>/<path...>"       -> tries frontend/<slug>/<path...> first
                                 (per-tenant override), then falls back
                                 to frontend/<path...> (the normal case)
  - "/api/..." (any depth)    -> 404, never served statically. API calls
                                 go to the Flask backend on its own port.
"""

import argparse
import http.server
import os
import posixpath
import socketserver
import urllib.parse


class SlugAwareRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serves files with the same slug-prefix fallback Flask uses in prod."""

    def _resolve_request_path(self, raw_path):
        """
        Returns the path (relative to the served directory, starting with
        '/') that should actually be served from disk, a string starting
        with '__REDIRECT__:' meaning "302 redirect there instead", or None
        if the request should be rejected with a 404.
        """
        parsed = urllib.parse.urlsplit(raw_path)
        clean_path = posixpath.normpath(urllib.parse.unquote(parsed.path))
        segments = [seg for seg in clean_path.split('/') if seg not in ('', '.')]

        # Never serve /api/* statically - that belongs to the Flask backend.
        if segments and segments[0].lower() == 'api':
            return None

        if not segments:
            return '/index.html'

        # Single segment: either a real top-level file ("/dashboard.html")
        # or a bare tenant slug with no extension ("/nhitmenv").
        if len(segments) == 1:
            if '.' in segments[0]:
                return f'/{segments[0]}'
            return '__REDIRECT__:/index.html'

        slug, *rest = segments
        filename = '/'.join(rest)

        if filename.lower() == 'index.html':
            return '__REDIRECT__:/index.html'

        # Prefer a genuine per-tenant override if one exists on disk...
        tenant_override = os.path.join(self.directory, slug, *rest)
        if os.path.isfile(tenant_override):
            return f'/{slug}/{filename}'

        # ...otherwise fall through to the real shared file, ignoring slug.
        fallback = os.path.join(self.directory, *rest)
        if os.path.isfile(fallback):
            return f'/{filename}'

        return None

    def _route(self):
        """Rewrites self.path in place if the request should be served.
        Sends the (404 or redirect) response itself and returns False if
        nothing further needs to happen."""
        resolved = self._resolve_request_path(self.path)

        if resolved is None:
            self.send_error(404, 'File not found')
            return False

        if resolved.startswith('__REDIRECT__:'):
            target = resolved.split(':', 1)[1]
            self.send_response(302)
            self.send_header('Location', target)
            self.end_headers()
            return False

        self.path = resolved
        return True

    def do_GET(self):
        if self._route():
            http.server.SimpleHTTPRequestHandler.do_GET(self)

    def do_HEAD(self):
        if self._route():
            http.server.SimpleHTTPRequestHandler.do_HEAD(self)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--directory', default='frontend')
    parser.add_argument('--bind', default='127.0.0.1')
    args = parser.parse_args()

    directory = os.path.abspath(args.directory)

    def handler_factory(*handler_args, **handler_kwargs):
        return SlugAwareRequestHandler(*handler_args, directory=directory, **handler_kwargs)

    with socketserver.TCPServer((args.bind, args.port), handler_factory) as httpd:
        print(f"Serving '{directory}' with slug-aware routing on http://{args.bind}:{args.port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
