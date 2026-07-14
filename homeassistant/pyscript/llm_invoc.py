import sys
import importlib


@service(supports_response="optional")
def answer_question(question=None, **kwargs):
    """Zentraler Alexa-Intent-Handler: entscheidet selbst über Tool-Nutzung."""
    # kwargs fängt z.B. return_response aus den Entwickler-Tools ab.

    # Always reload the helper modules so changes in python_scripts/ are picked
    # up without requiring a manual pyscript reload.
    sys.path.insert(0, "/config/python_scripts")
    import llm_handlers
    importlib.reload(llm_handlers)
    import llm_answer
    importlib.reload(llm_answer)
    from llm_answer import answer

    return task.executor(answer, question, hass)
