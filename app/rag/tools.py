"""Built-in tools for the agent pipeline.

Tools
-----
- web_search   – Internet search via DuckDuckGo (no API key required)
- weather      – Current weather via wttr.in (no API key required)
- datetime     – Current date, time, and timezone
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
from duckduckgo_search import DDGS

from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Web search ────────────────────────────────────────────────────────────────


def web_search(query: str, max_results: int = 5) -> str:
    """Run a DuckDuckGo text search and return formatted results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No web results found."
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            lines.append(f"[{i}] {title}\n{body}\nURL: {href}")
        return "\n\n".join(lines)
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        return f"Web search failed: {exc}"


# ── Weather ───────────────────────────────────────────────────────────────────


def get_weather(location: str) -> str:
    """Fetch current weather for a location via wttr.in."""
    try:
        clean = location.strip().replace(" ", "+")
        resp = httpx.get(
            f"https://wttr.in/{clean}",
            params={"format": "j1"},
            timeout=10,
            follow_redirects=True,
        )
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current_condition", [{}])[0]
        area = data.get("nearest_area", [{}])[0]
        city = area.get("areaName", [{}])[0].get("value", location)
        country = area.get("country", [{}])[0].get("value", "")
        desc = current.get("weatherDesc", [{}])[0].get("value", "")
        temp_c = current.get("temp_C", "?")
        temp_f = current.get("temp_F", "?")
        feels_c = current.get("FeelsLikeC", "?")
        humidity = current.get("humidity", "?")
        wind_kmph = current.get("windspeedKmph", "?")
        wind_dir = current.get("winddir16Point", "")
        visibility = current.get("visibility", "?")
        uv = current.get("uvIndex", "?")

        return (
            f"Weather for {city}, {country}:\n"
            f"  Condition: {desc}\n"
            f"  Temperature: {temp_c}°C ({temp_f}°F), feels like {feels_c}°C\n"
            f"  Humidity: {humidity}%\n"
            f"  Wind: {wind_kmph} km/h {wind_dir}\n"
            f"  Visibility: {visibility} km\n"
            f"  UV Index: {uv}"
        )
    except Exception as exc:
        logger.warning("Weather fetch failed for %r: %s", location, exc)
        return f"Could not fetch weather for '{location}': {exc}"


# ── Date / Time ───────────────────────────────────────────────────────────────


def get_current_datetime() -> str:
    """Return the current date, time, day of week, and timezone."""
    now = datetime.now(UTC)
    return (
        f"Current date and time (UTC):\n"
        f"  Date: {now.strftime('%A, %B %d, %Y')}\n"
        f"  Time: {now.strftime('%H:%M:%S %Z')}\n"
        f"  ISO:  {now.isoformat()}\n"
        f"  Unix: {int(now.timestamp())}"
    )


# ── Tool registry ────────────────────────────────────────────────────────────

TOOL_DESCRIPTIONS: dict[str, str] = {
    "rag": "Search uploaded documents to answer questions about their content",
    "web_search": (
        "Search the internet for current information, news, facts,"
        " or anything not in uploaded documents"
    ),
    "weather": "Get current weather conditions for a specific location",
    "datetime": "Get the current date, time, day of week",
    "general": "Answer general knowledge questions, greetings, or casual conversation directly",
}


# ── Gemini function declarations for native tool calling ─────────────────────

def get_gemini_tool_declarations():
    """Build google.genai Tool objects for Gemini function calling.

    Lazy import to avoid hard-coupling the module to google-genai at
    import time.
    """
    from google.genai import types

    declarations = [
        types.FunctionDeclaration(
            name="search_documents",
            description=(
                "DEFAULT tool. Search the user's uploaded documents "
                "(PDFs, DOCX, text files) to answer ANY question that "
                "could be answered from their content. Use this for "
                "questions about the user's data, personal info, names, "
                "projects, reports, or ANY factual question that is not "
                "explicitly about weather, current date/time, or web "
                "search. When in doubt, use this tool."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="The search query to find in documents",
                    ),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="web_search",
            description=(
                "Search the internet for real-time information, breaking "
                "news, or facts the user explicitly asks to look up "
                "online. Only use when the user says 'search the web', "
                "'look up online', or asks about very recent events."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="The web search query",
                    ),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_weather",
            description=(
                "Get current weather conditions for a specific location. "
                "Only use when the user explicitly asks about weather, "
                "temperature, or climate conditions for a place."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "location": types.Schema(
                        type=types.Type.STRING,
                        description="City or location, e.g. 'London'",
                    ),
                },
                required=["location"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_current_datetime",
            description=(
                "Get the current date, time, day of week, or year. "
                "Use when the user asks: what is today's date, what "
                "time is it, what day is it, what is the current date, "
                "what year is it, or similar date/time questions."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="direct_answer",
            description=(
                "ONLY for simple greetings (hi, hello, thanks, bye) "
                "and casual small talk. Do NOT use for any factual "
                "question, personal question, or information request."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "topic": types.Schema(
                        type=types.Type.STRING,
                        description="The greeting or small talk topic",
                    ),
                },
            ),
        ),
    ]
    return [types.Tool(function_declarations=declarations)]


# Map from function declaration names → graph node tool_name
FUNCTION_TO_TOOL: dict[str, str] = {
    "search_documents": "rag",
    "web_search": "web_search",
    "get_weather": "weather",
    "get_current_datetime": "datetime",
    "direct_answer": "general",
}
