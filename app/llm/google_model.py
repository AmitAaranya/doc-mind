from collections.abc import Iterator

from google import genai
from google.genai import types

from app.core import SETTING
from app.core.logging import get_logger

logger = get_logger(__name__)


class GoogleLLMModel:
    """wrapper around the Google GenAI streaming content API."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model or SETTING.GOOGLE_GENAI_MODEL
        self.api_key = api_key or SETTING.GOOGLE_CLOUD_API_KEY or SETTING.LLM_API_KEY
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Missing Google API key. Set GOOGLE_CLOUD_API_KEY or LLM_API_KEY."
                )

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
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
                ),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ],
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        )

        logger.debug("Starting streamed generation with model=%s", self.model)
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                yield chunk.text

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
