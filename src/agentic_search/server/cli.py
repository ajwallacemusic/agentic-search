"""`agentic-search` command line: run the service, or export the client contract."""

from __future__ import annotations

import argparse
import asyncio
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentic-search")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the HTTP/SSE search service")
    serve.add_argument("--config", required=True, help="service YAML (service: + profiles:)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--log-level", default="info")
    serve.add_argument("--graceful-timeout", type=float, default=10.0,
                       help="seconds to let open searches finish on shutdown (default 10)")
    export = sub.add_parser("export-schema", help="write the event JSON Schema and fixtures")
    export.add_argument("--out", default="schema")
    args = parser.parse_args(argv)
    try:
        import uvicorn

        from agentic_search.config import ConfigError
        from agentic_search.server import create_app, load_service
        from agentic_search.server.schema import export as export_schema
    except ImportError as exc:
        print(f"agentic-search: the server extra is not installed ({exc}); "
              "install with `pip install 'agentic-search[server]'`", file=sys.stderr)
        return 2
    if args.command == "serve":
        try:
            config, profiles = load_service(args.config)
            app = create_app(config, profiles)
        except ConfigError as exc:
            print(f"agentic-search: {exc}", file=sys.stderr)
            return 2
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level,
                    timeout_graceful_shutdown=args.graceful_timeout)
        return 0
    for path in asyncio.run(export_schema(args.out)):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
