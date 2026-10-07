"""Run the RASVC-X API server:  python -m rasvcx

This is the application entrypoint.  It is the one place that reads a
local `.env` file (so secrets such as RASVCX_LLM_API_KEY never have to be
exported by hand) and then starts Uvicorn with a single worker.

A single worker is required: the knowledge-base lease tracker, the
ingestion job store and the embedded/in-memory Qdrant modes are all
per-process state.

Mode selection (see rasvcx.config.loader.resolve_config_path):
    default                               -> config/research_hybrid.yaml
    RASVCX_CONFIG=<file>                  -> that file
    RASVCX_EXECUTION_MODE=offline_test    -> offline test doubles (explicit)

Environment variables already set in the process take precedence over
values in `.env`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rasvcx", description=__doc__)
    parser.add_argument("--host", default=None, help="bind host (default: api.host)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default: api.port)")
    parser.add_argument("--env-file", default=".env", help="dotenv file to load (default: .env)")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    if args.env_file and os.path.isfile(args.env_file):
        try:
            from dotenv import load_dotenv
        except ImportError:
            print(
                f"{args.env_file} exists but python-dotenv is not installed; "
                "install it or export the variables yourself.",
                file=sys.stderr,
            )
            return 2
        load_dotenv(args.env_file, override=False)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    from rasvcx.config.loader import load_settings
    from rasvcx.config.settings import ConfigurationError

    try:
        settings = load_settings()
    except (ConfigurationError, FileNotFoundError) as exc:
        print(f"RASVC-X cannot start: {exc}", file=sys.stderr)
        return 2

    import uvicorn

    from rasvcx.api.main import create_app

    uvicorn.run(
        create_app(settings=settings),
        host=args.host or settings.api.host,
        port=args.port or settings.api.port,
        workers=1,
        log_level=args.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
