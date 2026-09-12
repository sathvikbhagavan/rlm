# Annotator instructions

## Purpose and deadline

RxnHaystack tests reasoning over a cleaned 2023 USPTO reaction collection. Your
answers will help us measure human performance and find ambiguous questions or
incorrect/incomplete ground truths.

Work through the questions allocated by Amin. Each question shows a suggested
stopping limit; if you reach it without an answer, report “Suggested time exceeded
/ not found.” Unless Amin gives you a different date, return your ZIP by
**September 21, 2026 AoE**. Send questions and the completed file directly to
**Amin** through the channel he specifies.

## Before starting

1. Obtain the repository and assigned study link or question list from Amin. Follow
   `human_eval/STANDALONE_SETUP.md` for a first-time installation. If it is already
   installed, update the application before starting or resuming:

   ```bash
   git switch main
   git pull --ff-only
   ```

2. From the repository root, run `./human_eval/run.sh`, then open
   <http://127.0.0.1:8765> in the same computer's browser.
3. Optionally complete the non-identifying expertise profile. Do not enter your
   name, institution, or email in the application.
4. Use the assigned-study link if Amin supplied one; otherwise open **Benchmark
   questions** and work on the agreed questions.

## Answering a question

1. Derive your answer without viewing the stored answer. You may use local scripts,
   RDKit, spreadsheets, notebooks, literature/web search, LLMs or coding agents,
   offline processing, or another person.
2. **Inspect and verify the method or logic yourself. You remain responsible for
   the submitted judgment.** Record the tool categories used and separately report
   offline/tool-use minutes.
3. Enter your answer, confidence, and any useful rationale, then select **Submit**.
4. The page then shows the stored reference answer. Compare it with yours. If they
   differ and you believe your result is correct, select the ground-truth
   disagreement checkbox, explain the issue in the comments, and select **Submit
   revision**. This flags the item for author review while preserving both versions.

The first submitted answer remains the independent human-baseline answer. Later
revisions never overwrite it invisibly.

## Reusing code across questions

Related questions are intentionally amenable to shared tooling. Reuse is allowed
and encouraged—for example, a single parser and cached RDKit property table can
answer many Tier-1/Tier-2 variants, while common template-matching or reaction-graph
code can support groups of Tier-3/Tier-4 questions. Check each predicate and output
individually; do not assume that superficially similar questions use identical
chemical definitions.

## Saving, pausing, restoring, and returning work

- Work is autosaved to local SQLite every 10 seconds and when the page becomes
  hidden or closes. Use **Save and pause** before a planned break; it confirms the
  save, pauses timing, and returns to the list.
- To continue on the same computer, rerun `./human_eval/run.sh`, open the same URL,
  and choose **Begin or resume**. Keep `human_eval/local_state/`.
- Updating the application does not erase saved work. If the interface warns that
  a saved question has changed, review that question again before continuing; the
  earlier submission and its version remain in the revision history and export.
- **Export** downloads a checkpoint ZIP. On a fresh installation, choose
  **Restore**, upload that ZIP, and continue.
- Abrupt power loss can still lose typing since the most recent confirmed save. Use
  **Save and pause** periodically for long answers.
- When finished, select **Export** and send the single
  `rxnhaystack_human_annotation_<anonymous-id>.zip` to Amin without editing it.
