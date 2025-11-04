import os


def env(name: str) -> str:
    """Return required environment variable value or raise."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value
