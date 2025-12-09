(APIServer pid=1326221) INFO:     172.26.0.4:46890 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) INFO:     172.26.0.4:46874 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) WARNING 11-28 11:47:39 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4645 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4656 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) INFO:     172.26.0.4:46772 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) INFO:     172.26.0.4:50922 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) WARNING 11-28 11:47:39 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) WARNING 11-28 11:47:39 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4636 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4587 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) WARNING 11-28 11:47:39 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) INFO:     172.26.0.4:51106 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) INFO:     172.26.0.4:46898 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) WARNING 11-28 11:47:39 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) WARNING 11-28 11:47:39 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4578 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:39 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4584 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) INFO:     172.26.0.4:46890 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) INFO:     172.26.0.4:46874 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) WARNING 11-28 11:47:40 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) WARNING 11-28 11:47:40 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4581 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4647 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) INFO:     172.26.0.4:50922 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) INFO:     172.26.0.4:46772 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) WARNING 11-28 11:47:40 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) WARNING 11-28 11:47:40 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4645 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4650 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) INFO:     172.26.0.4:46898 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) INFO:     172.26.0.4:51106 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) WARNING 11-28 11:47:40 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) WARNING 11-28 11:47:40 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4655 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4649 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) INFO:     172.26.0.4:43508 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) INFO:     172.26.0.4:46874 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) WARNING 11-28 11:47:40 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4648 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4636 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) INFO:     172.26.0.4:46890 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) INFO:     172.26.0.4:50922 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) WARNING 11-28 11:47:40 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) WARNING 11-28 11:47:40 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4593 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4641 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) INFO:     172.26.0.4:46772 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) INFO:     172.26.0.4:51106 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) WARNING 11-28 11:47:40 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) WARNING 11-28 11:47:40 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) WARNING 11-28 11:47:40 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4582 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4640 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) INFO:     172.26.0.4:43508 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) INFO:     172.26.0.4:46874 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) WARNING 11-28 11:47:40 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) WARNING 11-28 11:47:40 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) WARNING 11-28 11:47:40 [protocol.py:126] The following fields were present in the request but ignored: {'repeat_penalty'}
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4650 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4653 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) INFO:     172.26.0.4:46890 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) INFO:     172.26.0.4:50922 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4650 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Error in preprocessing prompt inputs
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] Traceback (most recent call last):
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 235, in create_chat_completion
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     ) = await self._preprocess_chat(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1133, in _preprocess_chat
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 990, in _tokenize_prompt_input_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1011, in _tokenize_prompt_inputs_async
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 881, in _normalize_prompt_text_to_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]   File "/mnt/data/vllm-venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 962, in _validate_input
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257]     raise ValueError(
(APIServer pid=1326221) ERROR 11-28 11:47:40 [serving_chat.py:257] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4649 input tokens. Please reduce the length of the input messages.
(APIServer pid=1326221) INFO:     172.26.0.4:51106 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
(APIServer pid=1326221) INFO:     172.26.0.4:46772 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request
