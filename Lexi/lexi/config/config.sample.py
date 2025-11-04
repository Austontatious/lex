# config.sample.py
# Rename to config.py and customize as needed

DEVICE = "cuda"  # or "cpu"
MODEL_DIR = "/path/to/your/sd-models/"
AVATAR_OUTPUT_PATH = "./frontend/public/avatars/"

# Optional: Enable/disable avatar enhancement features
ENABLE_GFPGAN = True
ENABLE_UPSCALING = True

# Stable Diffusion prompt templates
BASE_PROMPT_STYLE = "best quality, masterpiece, 8k"
NEGATIVE_PROMPT = "poorly drawn, disfigured, low quality, bad anatomy"

# Memory configuration
MEMORY_PATH = "./memory/"
MAX_MEMORY_ITEMS = 200

# Web interface settings
FRONTEND_PORT = 5173
BACKEND_PORT = 8000
