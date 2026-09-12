from rlm.codeact_core import (
    FINAL_ANSWER_REQUIRED,
    FINAL_ANSWER_ATTEMPTS,
    INDEX_CODEACT_SYSTEM_PROMPT,
    PRELOADED_LINES_REMINDER,
    _continuation_instruction,
    _answer_only_attempts_exhausted,
    _has_final_answer,
    _is_answer_only_turn,
    append_preloaded_lines_reminder,
)


def test_preloaded_lines_reminder_is_last_in_long_codeact_prompt() -> None:
    prompt = "<context>many reaction rows</context>\n<question>find routes</question>\n"

    completed = append_preloaded_lines_reminder(prompt)

    assert completed.endswith(PRELOADED_LINES_REMINDER)
    assert completed.count("<tool-data-reminder>") == 1
    assert "`lines`" in completed
    assert completed.index("<question>") < completed.index("<tool-data-reminder>")


def test_codeact_final_marker_accepts_task_specific_multiline_answers() -> None:
    response = "ANSWER: 12,34,56,78\n90,91,92,93"

    assert _has_final_answer(response)


def test_codeact_does_not_treat_unfinished_reasoning_as_an_answer() -> None:
    assert not _has_final_answer("THINK: I should inspect one more candidate.")


def test_index_prompt_and_last_turn_instruction_do_not_assume_one_answer_shape() -> None:
    normalized_prompt = " ".join(INDEX_CODEACT_SYSTEM_PROMPT.split())
    assert "exact output format requested in the question" in normalized_prompt
    assert "Do not write or request more code" in FINAL_ANSWER_REQUIRED


def test_last_allowed_action_forces_a_bounded_final_answer_turn() -> None:
    normal = "continue with code"

    assert (
        _continuation_instruction(
            iteration=7, max_iterations=8, normal_instruction=normal
        )
        == normal
    )
    assert (
        _continuation_instruction(
            iteration=8, max_iterations=8, normal_instruction=normal
        )
        == FINAL_ANSWER_REQUIRED
    )


def test_answer_only_correction_has_one_retry_and_a_hard_stop() -> None:
    max_iterations = 8

    assert _is_answer_only_turn(iteration=9, max_iterations=max_iterations)
    assert not _answer_only_attempts_exhausted(
        iteration=9, max_iterations=max_iterations
    )
    assert _answer_only_attempts_exhausted(
        iteration=max_iterations + FINAL_ANSWER_ATTEMPTS,
        max_iterations=max_iterations,
    )
