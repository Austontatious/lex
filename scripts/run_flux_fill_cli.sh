#!/usr/bin/env bash
# Run Flux Fill + Florence2 + SAM2 workflow end-to-end against a ComfyUI server.
# Uses /upload/image, patches workflow JSON node widgets, queues a prompt, polls
# until completion, then downloads outputs into OUTDIR.

set -euo pipefail

COMFY_URL="${COMFY_URL:-http://127.0.0.1:8188}"
WORKFLOW_JSON="${WORKFLOW_JSON:-/mnt/data/Ultimate Flux Fill Inpainting + Redux ft Florence2Flux Large.json}"
TEST_IMAGE="${TEST_IMAGE:-}"
FLORENCE_PHRASE="${FLORENCE_PHRASE:-hair}"       # hair | dress | shirt | jacket | pants | shoes
INPAINT_PROMPT="${INPAINT_PROMPT:-bright red hair}"
OUTDIR="${OUTDIR:-./comfy_out}"

if [[ ! -f "${WORKFLOW_JSON}" ]]; then
  echo "ERROR: workflow JSON not found: ${WORKFLOW_JSON}" >&2
  exit 1
fi

if [[ -z "${TEST_IMAGE}" ]]; then
  echo "ERROR: set TEST_IMAGE to a real image path." >&2
  exit 1
fi

if [[ ! -f "${TEST_IMAGE}" ]]; then
  echo "ERROR: TEST_IMAGE not found: ${TEST_IMAGE}" >&2
  exit 1
fi

mkdir -p "${OUTDIR}"

echo "== Upload test image to ComfyUI =="
UPLOAD_JSON="$(curl -sS -X POST \
  -F "image=@${TEST_IMAGE}" \
  -F "overwrite=true" \
  "${COMFY_URL}/upload/image")"
export UPLOAD_JSON

UPLOADED_NAME="$(python3 - <<'PY'
import json, os
print(json.loads(os.environ["UPLOAD_JSON"]).get("name", ""))
PY
)"
export UPLOADED_NAME
if [[ -z "${UPLOADED_NAME}" ]]; then
  echo "ERROR: upload did not return a name: ${UPLOAD_JSON}" >&2
  exit 1
fi
echo "Uploaded as: ${UPLOADED_NAME}"

echo "== Queue prompt =="

python3 - <<'PY'
import json, os, time, requests, urllib.parse, sys

COMFY_URL = os.environ["COMFY_URL"]
WF_PATH = os.environ["WORKFLOW_JSON"]
IMG_NAME = os.environ["UPLOADED_NAME"]
PHRASE = os.environ["FLORENCE_PHRASE"]
PROMPT = os.environ["INPAINT_PROMPT"]
OUTDIR = os.environ["OUTDIR"]

# Node IDs from the provided workflow
NODE_POS_PROMPT = 6           # CLIPTextEncode (positive)
NODE_LOAD_IMAGE_MAIN = 66     # LoadImage main inpaint
NODE_LOAD_IMAGE_STYLE = 54    # LoadImage style (optional)
NODE_FLORENCE_RUN = 67        # Florence2Run

with open(WF_PATH, "r", encoding="utf-8") as f:
    wf = json.load(f)

# Patch nodes

def patch_node(nid, fn):
    for n in wf.get("nodes", []):
        if n.get("id") == nid:
            fn(n)
            return True
    return False

if not patch_node(NODE_POS_PROMPT, lambda n: n["widgets_values"].__setitem__(0, PROMPT)):
    print(f"ERROR: node {NODE_POS_PROMPT} not found", file=sys.stderr)
    sys.exit(1)

if not patch_node(NODE_FLORENCE_RUN, lambda n: n["widgets_values"].__setitem__(0, PHRASE)):
    print(f"ERROR: node {NODE_FLORENCE_RUN} not found", file=sys.stderr)
    sys.exit(1)

for nid in (NODE_LOAD_IMAGE_MAIN, NODE_LOAD_IMAGE_STYLE):
    patch_node(nid, lambda n: (
        n["widgets_values"].__setitem__(0, IMG_NAME),
        len(n.get("widgets_values", [])) > 1 and n["widgets_values"].__setitem__(1, "image")
    ))

payload = {"prompt": wf}
r = requests.post(f"{COMFY_URL}/prompt", json=payload, timeout=60)
r.raise_for_status()
resp = r.json()
prompt_id = resp.get("prompt_id")
if not prompt_id:
    print("ERROR: no prompt_id returned", resp, file=sys.stderr)
    sys.exit(1)
print("PROMPT_ID", prompt_id)

hist_url = f"{COMFY_URL}/history/{prompt_id}"
files = None
for _ in range(300):
    hr = requests.get(hist_url, timeout=30)
    if hr.status_code == 200:
        h = hr.json()
        if prompt_id in h and "outputs" in h[prompt_id]:
            outputs = h[prompt_id]["outputs"]
            files = []
            for _, out in outputs.items():
                for img in out.get("images", []):
                    files.append(img)
            if files:
                break
    time.sleep(1)

if not files:
    print("ERROR: timed out waiting for outputs", file=sys.stderr)
    sys.exit(2)

print("FILES_JSON", json.dumps(files))

os.makedirs(OUTDIR, exist_ok=True)
for im in files:
    fn = im["filename"]; sub = im.get("subfolder",""); typ = im.get("type","output")
    url = f"{COMFY_URL}/view?filename={urllib.parse.quote(fn)}&subfolder={urllib.parse.quote(sub)}&type={urllib.parse.quote(typ)}"
    out_path = os.path.join(OUTDIR, fn)
    print("Downloading", url, "->", out_path)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)

print(f"DONE. Files in {OUTDIR}")
PY
