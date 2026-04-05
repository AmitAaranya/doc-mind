from collections.abc import Iterator
from time import sleep
from typing import Any

from google import genai
from google.genai import types

from app.core import SETTING
from app.core.logging import get_logger
from app.llm.base import BaseLLM

logger = get_logger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF = (1, 3, 5)  # seconds between retries
_RETRYABLE_CODES = {503, 429, 500}


def _extract_status_code(exc: Exception) -> int | None:
    """Best-effort extraction of an HTTP/gRPC status code from a Google API error."""
    # google.api_core.exceptions.ServiceUnavailable, etc.
    if hasattr(exc, "code"):
        c = exc.code
        return c if isinstance(c, int) else None
    # google-genai wraps errors with .status_code or nested .code
    if hasattr(exc, "status_code"):
        return exc.status_code
    # Fall back to string matching for "503" / "429" in the message
    msg = str(exc)
    for code in _RETRYABLE_CODES:
        if str(code) in msg:
            return code
    return None


class GoogleLLMModel(BaseLLM):
    """wrapper around the Google GenAI streaming content API."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model or SETTING.GOOGLE_GENAI_MODEL
        self.api_key = api_key or SETTING.GOOGLE_CLOUD_API_KEY
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self.api_key:
                raise ValueError("Missing Google API key. Set GOOGLE_CLOUD_API_KEY.")

            self._client = genai.Client(
                vertexai=True,
                api_key=self.api_key,
            )

        return self._client

    def stream_text(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 1,
        thinking_level: str = "LOW",
    ) -> Iterator[str]:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
            ],
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel(thinking_level)
            ),
        )

        logger.debug("Starting streamed generation with model=%s", self.model)
        for attempt in range(_MAX_RETRIES):
            try:
                for chunk in self.client.models.generate_content_stream(
                    model=self.model,
                    contents=contents,
                    config=config,
                ):
                    if chunk.text:
                        yield chunk.text
                return  # success — exit retry loop
            except Exception as exc:
                code = _extract_status_code(exc)
                if code in _RETRYABLE_CODES and attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_BACKOFF[attempt]
                    logger.warning(
                        "Retryable error (code=%s, attempt %d/%d), "
                        "retrying in %ds: %s",
                        code, attempt + 1, _MAX_RETRIES, wait, exc,
                    )
                    sleep(wait)
                else:
                    raise

    def describe_image(self, image_b64: str, mime_type: str = "image/png") -> str:
        """Send an inline image to Gemini and return a descriptive text response.

        Parameters
        ----------
        image_b64:
            Base64-encoded image bytes (as returned by ImageData.data_b64).
        mime_type:
            MIME type string, e.g. ``"image/png"`` or ``"image/jpeg"``.
        """
        import base64

        image_bytes = base64.b64decode(image_b64)
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    types.Part.from_text(
                        text=(
                            "Describe this image extracted from a PDF document. "
                            "Include: visual type (chart, photo, diagram, screenshot, etc.), "
                            "key data points, all visible text and labels, "
                            "axes or legend values if present, and any contextual information "
                            "that would help answer questions about this document. "
                            "Be factual and specific."
                        )
                    ),
                ],
            )
        ]
        config = types.GenerateContentConfig(
            system_instruction=(
                "You are a document analysis assistant. "
                "Describe PDF images for use in a retrieval-augmented generation (RAG) system."
            ),
            temperature=0.2,
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
            ],
        )
        logger.debug("Describing image with model=%s mime_type=%s", self.model, mime_type)
        description_parts: list[str] = []
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                description_parts.append(chunk.text)
        return "".join(description_parts)

    def health_check(self) -> tuple[bool, str]:
        """Send a tiny probe prompt and verify the model returns text."""
        try:
            response = ""
            for chunk in self.stream_text("Hi"):
                if chunk:
                    response += chunk
                    if response.strip():
                        return True, response.strip()
            return False, "Empty response from LLM"
        except Exception as exc:
            logger.warning("LLM health check failed: %s", exc)
            return False, str(exc)

    def call_tool(
        self,
        prompt: str,
        tools: list[types.Tool],
        *,
        system_instruction: str | None = None,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Use Gemini native function calling to pick a tool.

        Returns ``{"name": "<function_name>", "args": {...}}`` for the
        first function call the model emits.
        """
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            tools=tools,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                ),
            ),
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
            ],
        )
        logger.debug("Tool call with model=%s, %d tool(s)", self.model, len(tools))
        for attempt in range(_MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                break  # success
            except Exception as exc:
                code = _extract_status_code(exc)
                if code in _RETRYABLE_CODES and attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_BACKOFF[attempt]
                    logger.warning(
                        "Retryable tool-call error (code=%s, attempt %d/%d), "
                        "retrying in %ds: %s",
                        code, attempt + 1, _MAX_RETRIES, wait, exc,
                    )
                    sleep(wait)
                else:
                    raise
        # Extract the first function call from the response
        for part in response.candidates[0].content.parts:
            if part.function_call:
                fc = part.function_call
                return {
                    "name": fc.name,
                    "args": dict(fc.args) if fc.args else {},
                }
        # Fallback if model didn't produce a function call
        return {"name": "general", "args": {}}
