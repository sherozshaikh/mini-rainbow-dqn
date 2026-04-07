"""Launch the live demo server. Run with: python -m mini_rainbow.scripts.demo"""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start live DQN demo server")
    parser.add_argument(
        "--dqn-checkpoint",
        type=str,
        default="gdrive/checkpoints/stage1_dqn_best.pt",
        help="Path to DQN checkpoint",
    )
    parser.add_argument(
        "--rainbow-checkpoint",
        type=str,
        default="gdrive/checkpoints/stage2_rainbow_lite_best.pt",
        help="Path to Rainbow-Lite checkpoint",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu|cuda")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    # Pass config via environment (picked up by app.startup)
    os.environ["DQN_CHECKPOINT"] = args.dqn_checkpoint
    os.environ["RAINBOW_CHECKPOINT"] = args.rainbow_checkpoint
    os.environ["DEVICE"] = args.device

    from mini_rainbow.src.demo.app import app

    logger.info(f"Starting demo server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
