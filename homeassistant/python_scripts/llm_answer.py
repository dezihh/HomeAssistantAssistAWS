import aiohttp
import asyncio
import yaml

MODEL = "Deepseekv4Pro"

with open("/config/secrets.yaml") as f:
    API_KEY = yaml.safe_load(f).get("litellm_key", "")

LLM_URL = "https://hotel.ziegler-eu.de/litellm/v1/chat/completions"
TEXT_URL = "https://knx.ziegler-eu.de/helloworld.txt"

SYSTEM_PROMPT = (
    "Du bist ein Sprachassistent, der über Alexa vorgelesen wird. "
    "Antworte kurz, in ganzen Sätzen, ohne Aufzählungen oder Markdown. "
    "Nutze das passende Werkzeug, wenn die Frage sich auf Smart-Home-Daten "
    "oder auf die helloworld-Textdatei bezieht. "
    "Für Fragen zur helloworld-Textdatei MUSST du das Werkzeug get_meineseite "
    "verwenden und den zurückgegebenen Text UNVERÄNDERT wiedergeben, "
    "einschließlich aller SSML-Tags. "
    "Alle anderen Fragen beantwortest du direkt aus deinem Allgemeinwissen."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_meineseite",
            "description": "Ruft den aktuellen Inhalt der helloworld-Textdatei ab. Der Text enthält SSML-Tags und muss unverändert an Alexa weitergegeben werden.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_home_assistant",
            "description": (
                "Für Fragen zu Smart-Home-Entities, Sensorwerten, "
                "Automationen oder Geräte-Steuerung."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def answer(question, hass):
    """Synchrone Einstiegsfunktion, wird von pyscript per task.executor aufgerufen."""
    return asyncio.run(_answer(question, hass))


async def _answer(question, hass):
    try:
        async with aiohttp.ClientSession() as session:
            q_lower = question.lower() if question else ""
            if "helloworld" in q_lower or "meineseite" in q_lower:
                return {"answer": await _fetch_text(session)}

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]

            first = await _call_llm(session, messages)
            if not first or "choices" not in first:
                return {"answer": "Entschuldigung, ich konnte die Antwort gerade nicht ermitteln."}

            msg = first["choices"][0]["message"]
            tool_calls = msg.get("tool_calls")

            if tool_calls:
                tool_call = tool_calls[0]
                tool_name = tool_call["function"]["name"]
                allowed = {t["function"]["name"] for t in TOOLS}
                if tool_name not in allowed:
                    return {"answer": "Diese Anfrage kann ich nicht bearbeiten."}

                if tool_name == "ask_home_assistant":
                    tool_result = await _ask_ha(hass, question)
                    return {"answer": tool_result}

                # get_meineseite: Text unverändert weitergeben, SSML erhalten
                return {"answer": await _fetch_text(session)}

            direct = _text(first)
            return {"answer": direct} if direct else {"answer": "Entschuldigung, ich konnte die Antwort gerade nicht ermitteln."}

    except Exception as exc:
        return {"answer": f"Fehler bei der Verarbeitung: {exc}"}


async def _call_llm(session, messages, tools=True):
    body = {
        "model": MODEL,
        "max_tokens": 800,
        "messages": messages,
    }
    if tools:
        body["tools"] = TOOLS

    async with session.post(
        LLM_URL,
        headers={"x-api-key": API_KEY, "content-type": "application/json"},
        json=body,
        timeout=aiohttp.ClientTimeout(total=15),
    ) as resp:
        return await resp.json()


async def _fetch_text(session):
    async with session.get(
        TEXT_URL,
        timeout=aiohttp.ClientTimeout(total=5),
    ) as r:
        return await r.text()


async def _ask_ha(hass, question):
    coro = hass.services.async_call(
        "conversation",
        "process",
        service_data={"text": question, "language": "de"},
        blocking=True,
        return_response=True,
    )
    future = asyncio.run_coroutine_threadsafe(coro, hass.loop)
    result = await asyncio.wrap_future(future)
    return result["response"]["speech"]["plain"]["speech"]


def _text(response):
    try:
        return response["choices"][0]["message"].get("content", "")
    except (KeyError, IndexError, TypeError):
        return ""
