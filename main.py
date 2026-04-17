from loguru import logger

from src.env import env
from src.server import run_server


def main() -> None:
    logger.info("Loaded environment settings: {}", env)
    run_server()


if __name__ == "__main__":
    main()
