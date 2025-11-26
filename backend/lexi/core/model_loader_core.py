# model_loader_core.py (upgrade)
from __future__ import annotations
import os, json, time, logging, re, copy
from typing import Any, Dict, List, Optional, Union, Generator, Tuple
import requests
from requests.adapters import HTTPAdapter, Retry

log = logging.getLogger("lexi.model_loader")

DEFAULT_BASE = os.getenv("LLM_API_BASE", "http://host.docker.internal:8008/v1").rstrip("/")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "Lexi")
# Use (connect, read) timeouts to avoid long hangs on streaming
TIMEOUT_CONNECT = float(os.getenv("LLM_HTTP_CONNECT_TIMEOUT", "2.0"))
TIMEOUT_READ = float(os.getenv("LLM_HTTP_READ_TIMEOUT", "120.0"))
TIMEOUT: Tuple[float, float] = (TIMEOUT_CONNECT, TIMEOUT_READ)

# Optional global stop tokens (kept small; per-call override allowed)
DEFAULT_STOPS = [s for s in (os.getenv("LLM_STOP", "<|im_end|>") or "").split(",") if s]


# Retry policy for transient errors
def _build_session() -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.3,
        status_forcelist=(408, 409, 429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({"Accept": "application/json"})
    return sess


class ModelLoader:
    """
    Resilient wrapper around an OpenAI-compatible Chat Completions API (vLLM/OpenAI/OpenRouter-like).
    - Accepts str, list[{role,content}], or {'messages': [...]}
    - Returns structured results with text, usage, finish_reason, raw
    """

    def __init__(self) -> None:
        self.base_url = DEFAULT_BASE
        self.model = DEFAULT_MODEL
        self.session = _build_session()

        # Legacy surface expected by persona_core
        self.primary_type = "chat"
        self.models = {"chat": self}

        # Default generation params (can be overridden per call)
        self.default_params: Dict[str, Any] = {
            "temperature": float(os.getenv("LEX_TEMP", "0.9")),
            "top_p": float(os.getenv("LEX_TOP_P", "0.9")),
            "presence_penalty": float(os.getenv("LEX_PRESENCE", "0.6")),
            "frequency_penalty": float(os.getenv("LEX_FREQUENCY", "0.1")),
            "stop": DEFAULT_STOPS or ["<|im_end|>"],
            # Common extras (filled only if provided per-call)
            # "seed": 0,
            # "response_format": {"type": "json_object"},
            "repetition_penalty": 1.15,  # vLLM supports this
        }

        # Warm model list lazily
        self._models_cache: Optional[List[str]] = None

    # ---------- helpers ----------
    @staticmethod
    def _coerce_to_messages(
        payload: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if isinstance(payload, str):
            msgs: List[Dict[str, Any]] = [{"role": "user", "content": payload}]
        elif isinstance(payload, list):
            msgs = [copy.deepcopy(m) if isinstance(m, dict) else {"role": "user", "content": str(m)} for m in payload]
        elif isinstance(payload, dict) and isinstance(payload.get("messages"), list):
            msgs = [copy.deepcopy(m) if isinstance(m, dict) else {"role": "user", "content": str(m)} for m in payload["messages"]]
        else:
            msgs = []

        def _strip_tags(text: str) -> str:
            return re.sub(r"</?tool_(call|response)>", "", text)

        # Strip vestigial tool tags from content and tool_call arguments
        for m in msgs:
            if not isinstance(m, dict):
                continue
            if isinstance(m.get("content"), str):
                m["content"] = _strip_tags(m["content"])
            tool_calls = m.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function")
                    if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                        fn["arguments"] = _strip_tags(fn["arguments"])
        return msgs

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        t0 = time.time()
        r = self.session.post(url, json=body, timeout=TIMEOUT)
        dt = (time.time() - t0) * 1000
        if r.status_code >= 400:
            # Try to print compact error body for logs
            try:
                err = r.json()
            except Exception:
                err = {"text": r.text[:500]}
            try:
                body_preview = json.dumps(body, ensure_ascii=False)
            except Exception:
                body_preview = str(body)
            log.warning(
                "LLM POST %s %s -> %s in %.0f ms: %s | body=%s",
                path,
                body.get("model"),
                r.status_code,
                dt,
                err,
                body_preview[:20000],
            )
            r.raise_for_status()
        return r.json()

    def _safe_params(self, **overrides: Any) -> Dict[str, Any]:
        params = dict(self.default_params)
        # Map friendly 'repeat_penalty' to vLLM 'repetition_penalty'
        if "repeat_penalty" in overrides and "repetition_penalty" not in overrides:
            overrides["repetition_penalty"] = overrides.pop("repeat_penalty")
        # Remove Nones to avoid confusing some servers
        clean = {k: v for k, v in overrides.items() if v is not None}
        params.update(clean)
        return params

    # ---------- discovery ----------
    def list_models(self, use_cache: bool = True) -> List[str]:
        if use_cache and self._models_cache is not None:
            return self._models_cache
        try:
            out = self._post("/models", {})  # some servers require GET; try POST first for compat
        except requests.HTTPError:
            # Fallback GET
            url = f"{self.base_url}/models"
            r = self.session.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            out = r.json()
        data = out.get("data") or []
        names = sorted({m.get("id") for m in data if isinstance(m, dict) and m.get("id")})
        self._models_cache = names
        return names

    def health(self) -> bool:
        try:
            _ = self.list_models(use_cache=False)
            return True
        except Exception as e:
            log.warning("LLM health check failed: %s", e)
            return False

    # ---------- public API ----------
    def generate(
        self,
        payload: Union[str, Dict[str, Any], List[Dict[str, str]]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        # extras:
        seed: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Return dict: { 'text', 'usage', 'finish_reason', 'raw' }
        (raw = full provider response)
        """
        messages = self._coerce_to_messages(payload)
        # vLLM rejects "auto" unless launched with tool support; coerce to None
        if tool_choice == "auto":
            tool_choice = None

        params = self._safe_params(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            seed=seed,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )
        body: Dict[str, Any] = {"model": self.model, "messages": messages, **params}

        raw = self._post("/chat/completions", body)
        # OpenAI/vLLM shape:
        # { choices: [ { message: {role, content, tool_calls?}, finish_reason, ... } ], usage: {...} }
        choice = (raw.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        finish = choice.get("finish_reason")
        return {"text": text, "usage": raw.get("usage"), "finish_reason": finish, "raw": raw}

    def generate_stream(
        self,
        payload: Union[str, Dict[str, Any], List[Dict[str, str]]],
        **overrides: Any,
    ) -> Generator[str, None, Dict[str, Any]]:
        """
        Yield text deltas; when finished, return a summary dict via StopIteration.value:
        { 'text', 'usage', 'finish_reason' }
        """
        messages = self._coerce_to_messages(payload)
        params = self._safe_params(stream=True, **overrides)
        body: Dict[str, Any] = {"model": self.model, "messages": messages, **params}

        url = f"{self.base_url}/chat/completions"
        text_accum = []
        finish_reason = None
        usage = None

        with self.session.post(url, json=body, timeout=TIMEOUT, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                    ch = (obj.get("choices") or [{}])[0]
                    delta = ch.get("delta") or ch.get("message") or {}
                    if "content" in delta and delta["content"]:
                        chunk = delta["content"]
                        text_accum.append(chunk)
                        yield chunk
                    # Some servers include finish_reason/usage intermittently or at end
                    finish_reason = ch.get("finish_reason") or finish_reason
                    usage = obj.get("usage") or usage
                except Exception:
                    continue

        return {"text": "".join(text_accum), "usage": usage, "finish_reason": finish_reason}
