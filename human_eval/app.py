from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .candidates import public_candidates
from .dataset_browser import DatasetIndex
from .db import Store
from .exporting import export_filename, export_zip
from .restoring import restore_export
from .schema import exact_answer_match, normalize_structured_answer

VALID_MODES = {"baseline", "audit", "prospective"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def deterministic_audit_sample(
    value: Any, *, question_id: str, seed: int = 20270910, size: int = 25
) -> list[Any]:
    entries = value if isinstance(value, list) else [value]
    ranked = sorted(
        entries,
        key=lambda item: hashlib.sha256(
            f"{seed}:{question_id}:{json.dumps(item, sort_keys=True)}".encode()
        ).digest(),
    )
    return ranked[:size]


def reference_entries(value: Any, answer_type: str) -> list[Any]:
    if answer_type == "single_chain" or not isinstance(value, list):
        return [value]
    return value


def create_app(
    *,
    bundle_dir: str | Path,
    state_dir: str | Path,
    dataset_path: str | Path,
    allow_network: bool = False,
) -> FastAPI:
    package_dir = Path(__file__).resolve().parent
    bundle = Path(bundle_dir).resolve()
    state = Path(state_dir).resolve()
    dataset = Path(dataset_path).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    annotation_context = {
        key: manifest.get(key)
        for key in (
            "schema_version",
            "bundle_version",
            "questions_sha256",
            "dataset_sha256",
        )
    }
    # The protected component's digest identifies its revision without carrying
    # answer material or administrator-facing field names into annotator exports.
    annotation_context["protected_answers_sha256"] = manifest.get("admin_ground_truth_sha256")
    questions_list = load_jsonl(bundle / "questions.jsonl")
    questions = {x["question_id"]: x for x in questions_list}
    ground_truth = {
        x["question_id"]: x for x in load_jsonl(bundle / manifest["ground_truth_location"])
    }
    store = Store(state / "annotations.sqlite3")
    profile = store.profile()
    dataset_index = DatasetIndex(state / "dataset_index.sqlite3", dataset)
    csrf = secrets.token_urlsafe(24)
    app = FastAPI(title="RxnHaystack Human Validation", docs_url=None, redoc_url=None)
    app.state.config = {
        "bundle": bundle,
        "state": state,
        "dataset": dataset,
        "allow_network": allow_network,
    }
    app.state.store = store
    app.state.profile = profile
    app.state.questions = questions
    app.state.manifest = manifest
    app.state.csrf = csrf
    app.state.dataset_index = dataset_index
    templates = Jinja2Templates(directory=package_dir / "templates")
    app.mount("/static", StaticFiles(directory=package_dir / "static"), name="static")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        if (
            request.method not in {"GET", "HEAD", "OPTIONS"}
            and request.headers.get("x-rxh-csrf") != csrf
        ):
            return JSONResponse({"detail": "Invalid CSRF token"}, status_code=403)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    def context(request: Request, **values: Any) -> dict[str, Any]:
        return {
            "request": request,
            "annotator_id": profile["annotator_id"],
            "csrf": csrf,
            "candidate_count": len(public_candidates(store)),
            **values,
        }

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        progress = {mode: store.progress(profile["annotator_id"], mode) for mode in VALID_MODES}
        return templates.TemplateResponse(
            request,
            "index.html",
            context(
                request,
                progress=progress,
                question_count=len(questions),
                candidate_count=len(public_candidates(store)),
            ),
        )

    @app.get("/profile", response_class=HTMLResponse)
    def profile_page(request: Request):
        expertise = json.loads(store.profile()["expertise_json"])
        return templates.TemplateResponse(
            request, "profile.html", context(request, expertise=expertise)
        )

    @app.get("/restore", response_class=HTMLResponse)
    def restore_page(request: Request):
        return templates.TemplateResponse(request, "restore.html", context(request))

    @app.post("/api/restore")
    async def restore(request: Request):
        form = await request.form()
        upload = form.get("archive")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(422, "RxnHaystack export ZIP required")
        content = await upload.read()
        try:
            result = restore_export(store, content)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        profile.clear()
        profile.update(store.profile())
        app.state.profile = profile
        return result

    @app.post("/api/profile")
    async def save_profile(request: Request):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(422, "Object expected")
        store.save_expertise(profile["annotator_id"], payload)
        return {"saved": True}

    @app.get("/questions", response_class=HTMLResponse)
    def question_list(
        request: Request,
        mode: str = "baseline",
        tier: str = "",
        category: str = "",
        search: str = "",
        status: str = "",
        study_id: str = "",
        filter_error: str = "",
    ):
        filter_errors = []
        if mode not in {"baseline", "audit"}:
            filter_errors.append("Unknown question mode; showing benchmark questions instead.")
            mode = "baseline"
        if tier and tier not in {"1", "2", "3", "4"}:
            filter_errors.append("The tier filter was invalid and has been reset to All.")
        tier_number = int(tier) if tier in {"1", "2", "3", "4"} else None
        valid_categories = {q["category"] for q in questions_list}
        if category and category not in valid_categories:
            filter_errors.append("The category filter was invalid and has been reset to All.")
        category = category if category in valid_categories else ""
        if status and status not in {"not-started", "started", "completed"}:
            filter_errors.append("The status filter was invalid and has been reset to All.")
        status = status if status in {"not-started", "started", "completed"} else ""
        if filter_error:
            filter_errors.append("The filter request was invalid; safe defaults were applied.")
        assigned = store.assigned_items(profile["annotator_id"], study_id, mode) if study_id else []
        assigned_positions = {item_id: position for position, item_id in enumerate(assigned)}
        source_rows = sorted(
            questions_list,
            key=lambda q: assigned_positions.get(q["question_id"], len(assigned_positions)),
        )
        rows = []
        for q in source_rows:
            if study_id and q["question_id"] not in assigned_positions:
                continue
            if tier_number and q["tier"] != tier_number:
                continue
            if category and q["category"] != category:
                continue
            if (
                search
                and search.lower() not in (q["question_id"] + " " + q["canonical_prompt"]).lower()
            ):
                continue
            draft = store.draft(profile["annotator_id"], mode, q["question_id"])
            qrow = {
                **q,
                "status": "completed"
                if draft["submitted_at"]
                else ("started" if draft["updated_at"] else "not-started"),
            }
            if status and qrow["status"] != status:
                continue
            rows.append(qrow)
        categories = sorted(valid_categories)
        return templates.TemplateResponse(
            request,
            "questions.html",
            context(
                request,
                rows=rows,
                mode=mode,
                tier=tier_number,
                category=category,
                search=search,
                status=status,
                study_id=study_id,
                categories=categories,
                filter_errors=filter_errors,
            ),
        )

    @app.get("/question/{question_id}", response_class=HTMLResponse)
    def question_detail(
        request: Request,
        question_id: str,
        mode: str = "baseline",
        study_id: str = "",
    ):
        if mode not in {"baseline", "audit"}:
            raise HTTPException(400, "Invalid question mode")
        q = questions.get(question_id)
        if q is None:
            raise HTTPException(404, "Question not found")
        draft = store.draft(profile["annotator_id"], mode, question_id)
        prefill = None
        if mode == "baseline" and not draft["updated_at"]:
            same_type_ids = [
                candidate["question_id"]
                for candidate in questions_list
                if (
                    candidate["tier"],
                    candidate["category"],
                    candidate["subcategory"],
                )
                == (q["tier"], q["category"], q["subcategory"])
            ]
            source = store.latest_submitted_payload(
                profile["annotator_id"],
                mode,
                same_type_ids,
                exclude_item_id=question_id,
            )
            if source:
                labels = {
                    "confidence": "confidence",
                    "offline_minutes": "offline/tool-use minutes",
                    "tools": "tools used",
                }
                carried = {
                    key: source["payload"][key]
                    for key in labels
                    if source["payload"].get(key) not in (None, "", [])
                }
                if carried:
                    draft["payload"] = {
                        **draft["payload"],
                        **carried,
                        "prefill_source_question_id": source["item_id"],
                    }
                    prefill = {
                        "source_question_id": source["item_id"],
                        "fields": [labels[key] for key in carried],
                    }

        if study_id:
            ordered_ids = store.assigned_items(profile["annotator_id"], study_id, mode)
        else:
            ordered_ids = [candidate["question_id"] for candidate in questions_list]
        navigation = {
            "previous": None,
            "next": None,
            "position": None,
            "total": len(ordered_ids),
            "study_id": study_id,
        }
        if question_id in ordered_ids:
            position = ordered_ids.index(question_id)
            navigation.update(
                {
                    "previous": ordered_ids[position - 1] if position else None,
                    "next": ordered_ids[position + 1] if position + 1 < len(ordered_ids) else None,
                    "position": position + 1,
                }
            )
        audit_truth = None
        baseline_evaluation = None
        if mode == "audit":
            gt = ground_truth[question_id]
            entries = reference_entries(gt["representation"], q["answer_type"])
            audit_truth = {
                "total": len(entries),
                "sample": deterministic_audit_sample(gt["representation"], question_id=question_id),
                "sample_seed": manifest["audit_sampling"]["seed"],
                "sample_method": manifest["audit_sampling"]["method"],
                "relevant_reaction_indices": gt["relevant_reaction_indices"][:100],
            }
        elif draft["submitted_at"]:
            # Feedback is only reachable after an independent baseline submission.
            gt = ground_truth[question_id]
            entries = reference_entries(gt["representation"], q["answer_type"])
            audit_truth = {
                "total": len(entries),
                "sample": entries[:100],
                "shown": min(len(entries), 100),
                "truncated": len(entries) > 100,
                "post_submission": True,
            }
            if draft["payload"].get("abstention"):
                baseline_evaluation = {"status": "abstained"}
            else:
                submitted_entries = draft["payload"].get("answer_entries")
                if not isinstance(submitted_entries, list):
                    _, submitted_entries = normalize_structured_answer(
                        str(draft["payload"].get("answer_exact", ""))
                    )
                baseline_evaluation = {
                    "status": (
                        "correct"
                        if exact_answer_match(
                            submitted_entries, gt["representation"], q["answer_type"]
                        )
                        else "incorrect"
                    )
                }
        return templates.TemplateResponse(
            request,
            "question.html",
            context(
                request,
                question=q,
                mode=mode,
                draft=draft,
                audit_truth=audit_truth,
                baseline_evaluation=baseline_evaluation,
                prefill=prefill,
                navigation=navigation,
                study_id=study_id,
            ),
        )

    @app.get("/api/question/{question_id}")
    def question_api(question_id: str, mode: str = "baseline"):
        if question_id not in questions:
            raise HTTPException(404)
        if mode == "baseline":
            public = dict(questions[question_id])
            public.pop("ground_truth_ref", None)
            return public
        if mode == "audit":
            return {"question": questions[question_id], "ground_truth": ground_truth[question_id]}
        raise HTTPException(400)

    @app.post("/api/save/{mode}/{item_id}")
    async def save(mode: str, item_id: str, request: Request):
        if mode not in VALID_MODES:
            raise HTTPException(400, "Invalid mode")
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(422, "Object expected")
        prior = store.draft(profile["annotator_id"], mode, item_id)
        if mode == "baseline" and prior["submitted_at"]:
            payload["post_ground_truth_reveal_revision"] = True
        if "answer_exact" in payload:
            payload["answer_exact"], payload["answer_entries"] = normalize_structured_answer(
                str(payload["answer_exact"])
            )
        payload = normalize_payload(payload)
        store.save_draft(
            profile["annotator_id"],
            mode,
            item_id,
            payload,
            annotation_context=annotation_context,
        )
        return {"saved": True}

    @app.post("/api/submit/{mode}/{item_id}")
    async def submit(mode: str, item_id: str, request: Request):
        if mode not in VALID_MODES:
            raise HTTPException(400, "Invalid mode")
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(422, "Object expected")
        prior = store.draft(profile["annotator_id"], mode, item_id)
        if mode == "baseline" and prior["submitted_at"]:
            payload["post_ground_truth_reveal_revision"] = True
        if mode == "baseline" and not (
            payload.get("abstention") or str(payload.get("answer_exact", "")).strip()
        ):
            raise HTTPException(422, "Answer or abstention required")
        if mode == "baseline" and payload.get("tools") and not payload.get("verified_logic"):
            raise HTTPException(422, "Confirm that you inspected the tool-assisted logic")
        if mode == "audit" and not payload.get("severity"):
            raise HTTPException(422, "Audit severity required")
        if mode == "prospective" and payload.get("overall_label") not in {
            "plausible_alternative",
            "uncertain",
            "implausible",
            "cannot_assess",
        }:
            raise HTTPException(422, "Prospective label required")
        if "answer_exact" in payload:
            payload["answer_exact"], payload["answer_entries"] = normalize_structured_answer(
                str(payload["answer_exact"])
            )
        payload = normalize_payload(payload)
        store.save_draft(
            profile["annotator_id"],
            mode,
            item_id,
            payload,
            submit=True,
            annotation_context=annotation_context,
        )
        store.timer(profile["annotator_id"], mode, item_id, "pause", elapsed_seconds=0)
        result = {"submitted": True}
        if mode == "baseline":
            if payload.get("abstention"):
                result["evaluation"] = {"status": "abstained"}
            else:
                result["evaluation"] = {
                    "status": (
                        "correct"
                        if exact_answer_match(
                            payload.get("answer_entries", []),
                            ground_truth[item_id]["representation"],
                            questions[item_id]["answer_type"],
                        )
                        else "incorrect"
                    )
                }
        return result

    @app.post("/api/timer/{mode}/{item_id}/{action}")
    async def timer(mode: str, item_id: str, action: str, request: Request):
        if mode not in VALID_MODES or action not in {"start", "heartbeat", "pause"}:
            raise HTTPException(400)
        payload = await request.json()
        return store.timer(
            profile["annotator_id"],
            mode,
            item_id,
            action,
            elapsed_seconds=float(payload.get("elapsed_seconds", 0)),
        )

    @app.get("/api/annotation-version/{mode}/{item_id}")
    def annotation_version(mode: str, item_id: str):
        if mode not in VALID_MODES:
            raise HTTPException(400, "Invalid mode")
        draft = store.draft(profile["annotator_id"], mode, item_id)
        if not draft["updated_at"]:
            return {"changed": False}
        prior = draft["annotation_context"]
        compared_fields = {
            "bundle_version",
            "questions_sha256",
            "protected_answers_sha256",
            "dataset_sha256",
        }
        changed = not prior or any(
            prior.get(key) != annotation_context.get(key) for key in compared_fields
        )
        return {
            "changed": changed,
            "message": (
                "This question or its stored answer changed after your response was saved. "
                "Your earlier work remains preserved; re-check the current definition before revising."
                if changed
                else ""
            ),
        }

    @app.post("/api/attachment/{mode}/{item_id}")
    async def upload_attachment(mode: str, item_id: str, request: Request):
        if mode not in VALID_MODES:
            raise HTTPException(400, "Invalid mode")
        form = await request.form()
        upload = form.get("attachment")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(422, "Attachment required")
        content = await upload.read()
        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(413, "Attachment limit is 5 MiB")
        try:
            return store.add_attachment(
                profile["annotator_id"],
                mode,
                item_id,
                upload.filename or "attachment.txt",
                upload.content_type or "application/octet-stream",
                content,
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/dataset", response_class=HTMLResponse)
    def dataset_page(
        request: Request, page: int = 1, q: str = "", exact: bool = False, index: int | None = None
    ):
        if not dataset_index.database.exists():
            raise HTTPException(503, "Dataset index not built; run the index-dataset command")
        result = dataset_index.query(page=page, search=q, exact=exact, index=index)
        return templates.TemplateResponse(
            request,
            "dataset.html",
            context(request, result=result, q=q, exact=exact, index=index, manifest=manifest),
        )

    @app.get("/download/dataset")
    def download_dataset():
        # This is the only filesystem dataset path exposed; no user path is accepted.
        return FileResponse(dataset, filename=dataset.name, media_type="text/plain")

    @app.get("/depict/{reaction_index}.png")
    def depict(reaction_index: int):
        if not dataset_index.database.exists():
            raise HTTPException(503, "Dataset index not built")
        rows = dataset_index.query(index=reaction_index)["rows"]
        if not rows:
            raise HTTPException(404, "Reaction not found")
        try:
            import io

            from rdkit.Chem import Draw, rdChemReactions

            reaction = rdChemReactions.ReactionFromSmarts(rows[0]["raw"], useSmiles=True)
            image = Draw.ReactionToImage(reaction, subImgSize=(260, 180))
            output = io.BytesIO()
            image.save(output, format="PNG")
        except Exception as error:
            raise HTTPException(422, f"RDKit depiction failed: {error}") from error
        return Response(output.getvalue(), media_type="image/png")

    @app.get("/download/evidence/{mode}/{item_id}")
    def download_evidence(mode: str, item_id: str, format: str = "csv"):
        if format not in {"csv", "jsonl"}:
            raise HTTPException(400, "Format must be csv or jsonl")
        indices: list[int]
        if mode == "audit" and item_id in ground_truth:
            indices = deterministic_audit_sample(
                ground_truth[item_id]["relevant_reaction_indices"],
                question_id=f"download:{item_id}",
                size=1000,
            )
        elif mode == "prospective":
            item = next((x for x in public_candidates(store) if x["candidate_id"] == item_id), None)
            if item is None:
                raise HTTPException(404)
            indices = [int(x) for x in item.get("reaction_indices", [])]
        else:
            raise HTTPException(403, "Evidence download is restricted to revealing modes")
        data = dataset_index.export_rows(indices, format)
        return Response(
            data,
            media_type="text/csv" if format == "csv" else "application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="evidence_{item_id}.{format}"'},
        )

    @app.get("/api/evidence/audit/{item_id}")
    def audit_evidence_api(item_id: str):
        if item_id not in ground_truth:
            raise HTTPException(404)
        indices = ground_truth[item_id]["relevant_reaction_indices"]
        return {
            "total": len(indices),
            "reaction_indices": deterministic_audit_sample(
                indices, question_id=f"evidence:{item_id}", size=100
            ),
            "sampling_seed": manifest["audit_sampling"]["seed"],
        }

    @app.post("/download/records")
    async def download_records(request: Request):
        payload = await request.json()
        format_ = payload.get("format", "csv")
        if format_ not in {"csv", "jsonl"}:
            raise HTTPException(400, "Format must be csv or jsonl")
        try:
            indices = [int(x) for x in payload.get("indices", [])]
        except (TypeError, ValueError) as error:
            raise HTTPException(422, "Integer indices required") from error
        data = dataset_index.export_rows(indices, format_)
        return Response(
            data,
            media_type="text/csv" if format_ == "csv" else "application/x-ndjson",
            headers={
                "Content-Disposition": f'attachment; filename="rxnhaystack_records.{format_}"'
            },
        )

    @app.get("/prospective", response_class=HTMLResponse)
    def prospective_list(request: Request, study_id: str = ""):
        assigned = (
            store.assigned_items(profile["annotator_id"], study_id, "prospective")
            if study_id
            else []
        )
        positions = {item_id: position for position, item_id in enumerate(assigned)}
        rows = []
        for item in public_candidates(store):
            if study_id and item["candidate_id"] not in positions:
                continue
            draft = store.draft(profile["annotator_id"], "prospective", item["candidate_id"])
            rows.append({**item, "completed": bool(draft["submitted_at"])})
        if assigned:
            rows.sort(key=lambda item: positions[item["candidate_id"]])
        return templates.TemplateResponse(
            request,
            "prospective.html",
            context(request, rows=rows, study_id=study_id),
        )

    @app.get("/prospective/{candidate_id}", response_class=HTMLResponse)
    def prospective_detail(request: Request, candidate_id: str):
        item = next(
            (x for x in public_candidates(store) if x["candidate_id"] == candidate_id), None
        )
        if item is None:
            raise HTTPException(404)
        question = questions.get(item["question_id"])
        draft = store.draft(profile["annotator_id"], "prospective", candidate_id)
        return templates.TemplateResponse(
            request,
            "candidate.html",
            context(request, item=item, question=question, draft=draft, mode="prospective"),
        )

    @app.get("/export")
    def export():
        data = export_zip(store, profile["annotator_id"], manifest)
        return Response(
            data,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{export_filename(profile["annotator_id"])}"'
            },
        )

    return app


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    clean = dict(payload)
    if "issue_tags_text" in clean:
        clean["issue_tags"] = [
            value.strip() for value in str(clean["issue_tags_text"]).split(",") if value.strip()
        ]
    return clean
