import asyncio
import re


# German number formatting: 1234.5 -> 1.234,50
# Reimplementation of the Jinja2 gfmt macro used in intent_scripts.yaml

def gfmt(value, decimals=2):
    try:
        s = f"{float(value):,.{decimals}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return str(value)


def extract_conversation_speech(result):
    """Extract SSML or plain text from a HA conversation.process response."""
    speech = result.get("response", {}).get("speech", {})
    if "ssml" in speech and speech["ssml"].get("speech"):
        return speech["ssml"]["speech"]
    if "plain" in speech and speech["plain"].get("speech"):
        return speech["plain"]["speech"]
    return str(result)


async def _run_hass_service(hass, domain, service, data):
    """Run an async HA service call from our own asyncio event loop."""
    coro = hass.services.async_call(
        domain,
        service,
        service_data=data,
        blocking=True,
        return_response=True,
    )
    future = asyncio.run_coroutine_threadsafe(coro, hass.loop)
    return await asyncio.wrap_future(future)


async def handle_source_fetch(source_config):
    """Fetch a configured text source (e.g. helloworld.txt)."""
    import aiohttp

    url = source_config.get("source")
    if not url:
        return "Keine Quelle konfiguriert."

    async with aiohttp.ClientSession() as session:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=10),
            headers={"Accept": "text/plain, */*"},
        ) as r:
            text = await r.text()

    return text if source_config.get("preserve_format", False) else text.strip()


async def handle_house_status(hass, question):
    """Trigger the existing 'hausstatus' intent script in Home Assistant."""
    try:
        result = await _run_hass_service(
            hass,
            "conversation",
            "process",
            {"text": "hausstatus", "language": "de"},
        )
        return extract_conversation_speech(result)
    except Exception as exc:
        return f"Fehler beim Abrufen des Hausstatus: {exc}"


async def handle_intent_script(hass, intent_name, question):
    """Trigger an arbitrary intent script (e.g. 'bmw_netzladung_an')."""
    try:
        result = await _run_hass_service(
            hass,
            "conversation",
            "process",
            {"text": intent_name, "language": "de"},
        )
        return extract_conversation_speech(result)
    except Exception as exc:
        return f"Fehler beim Ausführen von {intent_name}: {exc}"


async def handle_fuel_price(hass, question):
    """Read the Nordöl fuel price sensor and format the answer."""
    entity_id = "sensor.nordoel_sieker_landstrasse_178_super_e10"
    state = hass.states.get(entity_id)

    if state and state.state not in ["unknown", "unavailable", "", None]:
        try:
            price = float(state.state)
            return f"Super E10 bei Nordöl kostet derzeit {gfmt(price, 3)} Euro."
        except ValueError:
            return f"Der Preissensor liefert einen ungültigen Wert: {state.state}."

    return "Der aktuelle Benzinpreis ist leider nicht verfügbar."
