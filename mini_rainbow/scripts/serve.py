"""Launch the FastAPI inference server. Run with: python -m mini_rainbow.scripts.serve"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from mini_rainbow.src.api.app import app, load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start DQN inference API server")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--num-actions", type=int, default=4, help="Number of actions")
    parser.add_argument("--dueling", action="store_true", help="Use dueling network")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu|cuda")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    # Load model before starting server
    load_model(
        checkpoint_path=args.checkpoint,
        num_actions=args.num_actions,
        dueling=args.dueling,
        device=args.device,
    )

    logger.info(f"Starting API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
