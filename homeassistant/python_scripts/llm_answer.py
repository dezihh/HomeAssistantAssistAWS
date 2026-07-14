import aiohttp
import asyncio
import json
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def _debug(msg):
    """Append a debug line to /config/llm_debug.log synchronously."""
    try:
        with open("/config/llm_debug.log", "a", encoding="utf-8") as f:
            from datetime import datetime
            f.write(f"{datetime.now().isoformat()} {msg}\n")
    except Exception:
        pass


from llm_handlers import (
    gfmt,
    handle_fuel_price,
    handle_house_status,
    handle_intent_script,
    handle_source_fetch,
)

MODEL = "Deepseekv4Pro"

LLM_URL = "https://hotel.ziegler-eu.de/litellm/v1/chat/completions"
SEARXNG_URL = "http://192.168.10.4:8003/search"

CONFIG_DIR = Path("/config/python_scripts")

API_KEY = None
LLM_SOURCES = None
LLM_ROUTES = None


def _load_secrets():
    global API_KEY
    if API_KEY is None:
        with open("/config/secrets.yaml") as f:
            API_KEY = yaml.safe_load(f).get("litellm_key", "")


def _load_configs():
    global LLM_SOURCES, LLM_ROUTES
    if LLM_SOURCES is None:
        with open(CONFIG_DIR / "llm_sources.json") as f:
            LLM_SOURCES = json.load(f)
    if LLM_ROUTES is None:
        with open(CONFIG_DIR / "llm_routes.json") as f:
            LLM_ROUTES = json.load(f)

SYSTEM_PROMPT = (
    "Du bist Smart Pilot, ein Sprachassistent für Home Assistant über Alexa. "
    "Antworte kurz, in ganzen Sätzen, ohne Aufzählungen oder Markdown. "
    "Nutze Werkzeuge, wenn die Frage danach verlangt:\n"
    "- Für Smart-Home-Abfragen (Temperatur, Verbrauch, Status): "
    "verwende find_ha_entities, dann get_ha_state.\n"
    "  Hinweis: Bei Thermostaten steht die aktuelle Raumtemperatur im Attribut "
    "'current_temperature' des get_ha_state Ergebnisses.\n"
    "- Für Geräte-Steuerung (Licht, Schalter, Rolladen, Klima): "
    "verwende control_device.\n"
    "- Für aktuelle Informationen aus dem Internet: verwende search_web.\n"
    "- Für Benzinpreise: verwende get_fuel_prices.\n"
    "- Für den Hausstatus: verwende get_house_status.\n"
    "- Für die helloworld-Textdatei: verwende get_meineseite.\n"
    "Falls keine Werkzeuge nötig sind, antworte direkt aus deinem Wissen. "
    "Zahlen sollen für Alexa verständlich ausgesprochen werden (z. B. '22,4 Grad')."
)

LLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_ha_entities",
            "description": (
                "Findet Home Assistant Entities anhand von Friendly Name oder entity_id. "
                "Verwende dies IMMER zuerst für Fragen zu Temperatur, Verbrauch, Sensorwerten, "
                "Status von Geräten oder Räumen."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Stichwort, z. B. 'Schlafzimmer Temperatur' oder 'Verbrauch'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ha_state",
            "description": (
                "Liest den aktuellen Zustand einer spezifischen Home Assistant Entity. "
                "Verwende dies, nachdem find_ha_entities die passende entity_id geliefert hat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Home Assistant entity_id, z. B. 'sensor.schlafzimmer_temperature'.",
                    }
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_device",
            "description": (
                "Schaltet, dimmt oder bewegt Home Assistant Geräte (Licht, Schalter, Rolladen, Klima). "
                "Erlaubte Aktionen: turn_on, turn_off, toggle, open_cover, close_cover, stop_cover, set_temperature."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Home Assistant entity_id, z. B. 'switch.wohnzimmer_lampe'.",
                    },
                    "action": {
                        "type": "string",
                        "enum": [
                            "turn_on",
                            "turn_off",
                            "toggle",
                            "open_cover",
                            "close_cover",
                            "stop_cover",
                            "set_temperature",
                        ],
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Nur für climate.set_temperature: Zieltemperatur in Grad Celsius.",
                    },
                },
                "required": ["entity_id", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Sucht im Internet über SearXNG, wenn aktuelle Informationen benötigt werden "
                "(z. B. Börsenkurse, Nachrichten, Wetterwarnungen)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Suchbegriff, z. B. 'DAX Stand heute'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fuel_prices",
            "description": "Holt den aktuellen Benzinpreis bei Nordöl (Super E10).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_house_status",
            "description": "Erstellt einen umfassenden Hausstatus-Bericht (Akkustand, Verbrauch, Solar, Benzinpreis).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_meineseite",
            "description": "Liest den Inhalt der helloworld-Textdatei (enthält SSML).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

ALLOWED_SERVICES = {
    "switch": ["turn_on", "turn_off", "toggle"],
    "light": ["turn_on", "turn_off", "toggle"],
    "cover": ["open_cover", "close_cover", "stop_cover"],
    "fan": ["turn_on", "turn_off", "toggle"],
    "climate": ["turn_on", "turn_off", "set_temperature"],
    "script": ["turn_on"],
    "automation": ["trigger", "turn_on", "turn_off"],
}


def answer(question, hass):
    """Synchrone Einstiegsfunktion, wird von pyscript per task.executor aufgerufen."""
    # Load secrets/configs here because this function runs in task.executor,
    # outside the HA event loop, avoiding blocking-call warnings.
    _load_secrets()
    _load_configs()
    return asyncio.run(_answer(question, hass))


async def _answer(question, hass):
    try:
        return await _answer_inner(question, hass)
    except Exception as exc:
        logger.error(f"Unhandled error in _answer: {exc}", exc_info=True)
        _debug(f"Unhandled error in _answer: {exc}")
        return {"answer": f"Entschuldigung, bei der Verarbeitung ist ein Fehler aufgetreten: {exc}"}


async def _answer_inner(question, hass):
    _debug(f"_answer called with: {question}")
    q_lower = question.lower() if question else ""

    # --- 1. deterministic content-source routing (SSML-safe) ---
    source_match = _find_best_match(q_lower, LLM_SOURCES)
    if source_match:
        return {"answer": await handle_source_fetch(source_match)}

    # --- 2. deterministic route routing ---
    route_match = _find_best_match(q_lower, LLM_ROUTES)
    if route_match:
        handler_name = route_match.get("handler", "")
        if handler_name == "house_status":
            return {"answer": await handle_house_status(hass, question)}
        elif handler_name == "fuel_price":
            return {"answer": await handle_fuel_price(hass, question)}
        elif handler_name.startswith("intent_script."):
            intent_name = handler_name.split(".", 1)[1]
            return {"answer": await handle_intent_script(hass, intent_name, question)}

    # --- 3. LLM with tools ---
    async with aiohttp.ClientSession() as session:

        async def find_ha_entities(query: str, max_results: int = 10):
            # Split query into terms and score matches; this catches entities
            # like 'climate.thermostat_schlafzimmer' for "Schlafzimmer Temperatur".
            terms = [t for t in query.lower().split() if len(t) >= 2]
            all_states = hass.states.async_all()
            scored = []
            for state in all_states:
                friendly_name = (
                    state.attributes.get("friendly_name", state.entity_id).lower()
                )
                entity_id = state.entity_id.lower()
                score = 0
                for term in terms:
                    if term in friendly_name:
                        score += 2
                    elif term in entity_id:
                        score += 1
                if score > 0:
                    scored.append((score, state))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = []
            for score, state in scored[:max_results]:
                results.append(
                    {
                        "entity_id": state.entity_id,
                        "name": state.attributes.get("friendly_name", state.entity_id),
                        "state": state.state,
                        "unit": str(state.attributes.get("unit_of_measurement", "")),
                    }
                )
            return results

        async def get_ha_state(entity_id: str):
            # hass.states has sync get(), async_all() exists for listing.
            state = hass.states.get(entity_id)
            if not state and "." in entity_id:
                eid_lower = entity_id.lower()
                for s in hass.states.async_all():
                    if s.entity_id.lower() == eid_lower:
                        state = s
                        break
            if not state:
                return {"error": f"Entity {entity_id} nicht gefunden."}
            attrs = dict(state.attributes)
            # keep only speech-relevant attributes
            relevant = {
                "current_temperature",
                "target_temperature",
                "temperature",
                "brightness",
                "position",
                "battery_level",
                "unit_of_measurement",
                "friendly_name",
                "hvac_mode",
                "fan_mode",
            }
            filtered = {k: str(attrs[k]) for k in relevant if k in attrs}
            result = {
                "entity_id": state.entity_id,
                "name": filtered.get("friendly_name", state.entity_id),
                "state": state.state,
                "unit": filtered.get("unit_of_measurement", ""),
                "attributes": filtered,
            }
            if "current_temperature" in filtered:
                result["current_temperature"] = filtered["current_temperature"]
            if "target_temperature" in filtered:
                result["target_temperature"] = filtered["target_temperature"]
            return result

        async def control_device(entity_id: str, action: str, temperature: float = None):
            if "." not in entity_id:
                return {
                    "error": "Ungültige entity_id. Erwartet wird z. B. 'switch.wohnzimmer'."
                }
            domain = entity_id.split(".")[0]
            if domain not in ALLOWED_SERVICES:
                return {"error": f"Domäne '{domain}' ist für Sprachsteuerung nicht erlaubt."}
            if action not in ALLOWED_SERVICES[domain]:
                return {"error": f"Aktion '{action}' für {domain} nicht erlaubt."}
            data = {"entity_id": entity_id}
            if action == "set_temperature" and temperature is not None:
                data["temperature"] = temperature
            try:
                await _hass_service(hass, domain, action, data)
                return {
                    "success": True,
                    "message": f"{action} auf {entity_id} ausgeführt.",
                }
            except Exception as exc:
                return {"error": str(exc)}

        async def search_web(query: str):
            params = {"q": query, "format": "json", "language": "de"}
            async with session.get(
                SEARXNG_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                data = await r.json()
            results = data.get("results", [])[:5]
            snippets = []
            for item in results:
                title = item.get("title", "").strip()
                content = item.get("content", "").strip()
                if title or content:
                    snippets.append(f"{title}: {content}")
            if not snippets:
                return "Keine Suchergebnisse gefunden."
            return "\n\n".join(snippets)

        async def get_fuel_prices():
            return await handle_fuel_price(hass, question)

        async def get_house_status():
            return await handle_house_status(hass, question)

        async def get_meineseite():
            return await handle_source_fetch(LLM_SOURCES["helloworld"])

        tool_impls = {
            "find_ha_entities": find_ha_entities,
            "get_ha_state": get_ha_state,
            "control_device": control_device,
            "search_web": search_web,
            "get_fuel_prices": get_fuel_prices,
            "get_house_status": get_house_status,
            "get_meineseite": get_meineseite,
        }

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        for iteration in range(5):
            response = await _call_llm(session, messages, tools=LLM_TOOLS)
            _debug(f"iteration {iteration}: {json.dumps(response, ensure_ascii=False)[:1500]}")
            if not response or "choices" not in response:
                logger.error(f"Invalid LLM response: {response}")
                return {
                    "answer": "Entschuldigung, ich konnte die Antwort gerade nicht ermitteln."
                }

            msg = response["choices"][0]["message"]
            tool_calls = msg.get("tool_calls")
            _debug(f"tool_calls: {tool_calls}")

            if not tool_calls:
                content = msg.get("content", "")
                return {
                    "answer": (
                        content.strip()
                        if content
                        else "Entschuldigung, ich hatte keine passende Antwort."
                    )
                }

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.get("content", ""),
                    "tool_calls": tool_calls,
                }
            )

            for tool_call in tool_calls:
                name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])
                impl = tool_impls.get(name)
                try:
                    raw = await impl(**args) if impl else "Werkzeug nicht verfügbar."
                    _debug(f"tool {name} result: {raw}")
                except Exception as exc:
                    raw = f"Fehler bei {name}: {exc}"
                    _debug(raw)
                    logger.error(raw)
                result_text = json.dumps(raw, ensure_ascii=False) if not isinstance(raw, str) else raw
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": name,
                        "content": result_text,
                    }
                )

        # Fallback after max iterations
        return {"answer": "Entschuldigung, diese Anfrage war zu komplex. Bitte formuliere sie einfacher."}


async def _hass_service(hass, domain, service, data):
    """Execute an async HA service call from llm_answer's own event loop."""
    coro = hass.services.async_call(
        domain,
        service,
        service_data=data,
        blocking=True,
        return_response=False,
    )
    future = asyncio.run_coroutine_threadsafe(coro, hass.loop)
    return await asyncio.wrap_future(future)


async def _call_llm(session, messages, tools=None):
    body = {
        "model": MODEL,
        "max_tokens": 800,
        "messages": messages,
    }
    if tools:
        body["tools"] = tools

    async with session.post(
        LLM_URL,
        headers={"x-api-key": API_KEY, "content-type": "application/json"},
        json=body,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        return await resp.json()


def _find_best_match(q_lower: str, config: dict):
    """Return the config item for the longest matching keyword."""
    best_match = None
    best_len = 0
    for key, value in config.items():
        key_lower = key.lower()
        if key_lower in q_lower and len(key_lower) > best_len:
            best_match = value
            best_len = len(key_lower)
    return best_match


# Keep gfmt importable for other callers
__all__ = ["answer"]
