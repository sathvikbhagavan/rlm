from rlm.codeact_core import PRELOADED_LINES_REMINDER, append_preloaded_lines_reminder


def test_preloaded_lines_reminder_is_last_in_long_codeact_prompt() -> None:
    prompt = "<context>many reaction rows</context>\n<question>find routes</question>\n"

    completed = append_preloaded_lines_reminder(prompt)

    assert completed.endswith(PRELOADED_LINES_REMINDER)
    assert completed.count("<tool-data-reminder>") == 1
    assert "`lines`" in completed
    assert completed.index("<question>") < completed.index("<tool-data-reminder>")
