#!/usr/bin/env python3
"""Run forecast: python scripts/run_forecast.py --domain-id X --state '...' --actions 'a,b,c'"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.decision.forecast import run_forecast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.domain.schemas import DecisionRequest
from app.services import decision_service


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain-id", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--actions", required=True, help="Comma-separated actions")
    p.add_argument("--provider", default=None)
    p.add_argument("--recommend", action="store_true")
    args = p.parse_args()
    body = DecisionRequest(
        domain_id=args.domain_id,
        context_state=args.state,
        candidate_actions=[a.strip() for a in args.actions.split(",") if a.strip()],
    )
    if args.recommend:
        result = await decision_service.recommend(body, provider=args.provider)
    else:
        result = {"forecast": await run_forecast(body.domain_id, body.context_state, body.candidate_actions, provider=args.provider)}
    print(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
