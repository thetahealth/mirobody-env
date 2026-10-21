"""Serve one SYNTHETIC question over HTTP or MCP; evaluation only, not medical advice."""
from __future__ import annotations

import argparse
import sys

from haenv.serve.session import load_question


def main(argv=None):
    parser = argparse.ArgumentParser(prog='haenv serve')
    sub = parser.add_subparsers(dest='transport', required=True)
    for name in ('fhir', 'mcp'):
        p = sub.add_parser(name)
        p.add_argument('--pack', required=True, help='Saved batch directory or pack manifest JSON')
        p.add_argument('--question', required=True, help='Question ID in payloads.jsonl')
        if name == 'fhir':
            p.add_argument('--port', type=int, default=8000)
            p.add_argument('--host', default='127.0.0.1')
            p.add_argument('--anchor', default='2000-01-01', help='ISO date for case day zero')
    args = parser.parse_args(argv)
    try:
        session = load_question(args.pack, args.question)
        if args.transport == 'mcp':
            from haenv.serve.mcp_server import run
            run(session)
        else:
            from haenv.serve.fhir_server import FHIRServer
            with FHIRServer((args.host, args.port), session, anchor=args.anchor) as server:
                print(f'SYNTHETIC FHIR session listening on port {server.server_port}', file=sys.stderr)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0
