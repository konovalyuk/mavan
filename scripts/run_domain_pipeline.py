#!/usr/bin/env python3
"""Start continuous domain conveyor: python scripts/run_domain_pipeline.py --domain-id <id> [--provider gemini]

Runs until Ctrl+C, then stops the conveyor.
"""
import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.domain_conveyor import start_conveyor, stop_conveyor


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain-id", required=True)
    p.add_argument("--provider", default=None)
    args = p.parse_args()

    result = await start_conveyor(args.domain_id, provider=args.provider)
    print(json.dumps(result, default=str, indent=2))
    print("Conveyor running. Press Ctrl+C to stop.", flush=True)

    stop = asyncio.Event()

    def _handle(*_):
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle)
        except NotImplementedError:
            pass

    await stop.wait()
    stopped = await stop_conveyor(args.domain_id)
    print(json.dumps(stopped, default=str, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
