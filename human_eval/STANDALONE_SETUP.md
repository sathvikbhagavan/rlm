# Standalone annotator setup

## Setup

1. Install `git` and [`uv`](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone the repository's current `main` branch and enter it:

   ```bash
   git clone https://github.com/sathvikbhagavan/rlm.git
   cd rlm
   ```

   If you already cloned it, update it before starting or resuming:

   ```bash
   git switch main
   git pull --ff-only
   ```

3. Launch:

   ```bash
   ./human_eval/run.sh
   ```

4. Open <http://127.0.0.1:8765>. The first launch installs isolated dependencies,
   builds and verifies the 100-question bundle, and indexes the bundled dataset.

Use **Save and pause** between sessions. Later, run the same launch command to
resume. When finished, click **Export** and send the single ZIP file to Amin.

## Independent-answer safeguard

The standalone application must carry protected answers locally in order to reveal
them after submission. Do not inspect ground-truth files or
`human_eval/generated/*/admin/`. Run coding assistants in a separate working
directory containing only a copy of `human_eval/data/reactionSmilesFigShareUSPTO2023_cleaned.txt`
and the annotator's own scripts; do not give them this repository as their working
directory.
