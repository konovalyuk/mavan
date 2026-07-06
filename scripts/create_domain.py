#!/usr/bin/env python3
"""Create a domain: python scripts/create_domain.py --name energy --description 'Energy policy'"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.domain.schemas import DomainCreate
from app.services import domain_service


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--description", default="")
    p.add_argument("--run-pipeline", action="store_true")
    p.add_argument("--provider", default=None)
    args = p.parse_args()
    domain = await domain_service.create_domain(DomainCreate(name=args.name, description=args.description))
    print(json.dumps(domain.model_dump(), default=str, indent=2))
    if args.run_pipeline:
        result = await domain_service.start_pipeline(domain.id, provider=args.provider)
        print(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
