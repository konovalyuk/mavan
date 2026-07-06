#!/usr/bin/env python3
"""Run domain pipeline: python scripts/run_domain_pipeline.py --domain-id <id> [--provider gemini]"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.domain.pipeline import run_domain_pipeline


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain-id", required=True)
    p.add_argument("--provider", default=None)
    args = p.parse_args()
    result = await run_domain_pipeline(args.domain_id, provider=args.provider)
    print(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
