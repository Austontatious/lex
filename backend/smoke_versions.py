from importlib.metadata import version


def assert_version(package: str, expected: str) -> None:
    actual = version(package)
    if actual != expected:
        raise AssertionError(f"{package} {actual} != {expected}")

def assert_optional(package: str, expected: str) -> None:
    try:
        actual = version(package)
    except Exception:
        return
    if actual != expected:
        raise AssertionError(f"{package} {actual} != {expected}")


assert_version("fastapi", "0.116.1")
assert_version("starlette", "0.47.3")
assert_version("pydantic", "2.11.9")
assert_version("httpx", "0.27.2")
assert_version("httpcore", "1.0.9")
assert_version("h11", "0.14.0")

assert_version("torch", "2.5.1+cu121")
assert_version("torchvision", "0.20.1+cu121")
assert_version("torchaudio", "2.5.1+cu121")

assert_version("transformers", "4.56.2")
assert_version("tokenizers", "0.21.0")
assert_version("huggingface-hub", "0.35.1")
assert_version("safetensors", "0.6.2")

assert_optional("xformers", "0.0.28.post3")

print("backend package versions OK")
