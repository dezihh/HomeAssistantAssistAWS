import sys
import importlib

sys.path.insert(0, "/config/python_scripts")
import llm_answer
importlib.reload(llm_answer)
from llm_answer import answer


@service(supports_response="optional")
def answer_question(question=None, **kwargs):
    """Zentraler Alexa-Intent-Handler: entscheidet selbst über Tool-Nutzung."""
    # kwargs fängt z.B. return_response aus den Entwickler-Tools ab.
    return task.executor(answer, question, hass)
