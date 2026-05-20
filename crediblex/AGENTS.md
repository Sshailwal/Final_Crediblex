# Agent Instructions for CredibleX

## Purpose
Help AI coding agents quickly understand CredibleX and make safe, effective changes.

## What this project is
- **CredibleX** is a Python + FastAPI backend with a React + Vite frontend.
- It analyzes news URLs or raw text and produces a credibility report with a trust score.
- The model is a multi-task PyTorch transformer built on `microsoft/deberta-v3-base`.

## Primary directories and files
- `api.py` — FastAPI server entrypoint exposing `/health`, `/analyze`, `/analyze-text`, and `/logs`.
- `inference.py` — model loading, prediction code, report generation, trust score logic.
- `model.py` — PyTorch model definition and task heads.
- `train.py` — training loop for the credibility model.
- `config.py` — hyperparameters, device selection, model save path, and training settings.
- `schema.py` — Pydantic schemas used across the project.
- `requirements.txt` — backend Python dependencies.
- `frontend/package.json` — frontend scripts and dependencies.
- `project_prompt.md` — project mission, model architecture, and AI guidance.

## Important behavior and conventions
- The backend expects a trained model checkpoint at the path defined by `config.SAVE_PATH`.
- `api.py` validates inputs and raises HTTP errors with structured details.
- `inference.py` returns a final report with:
  - `score` (0–100)
  - `verdict` (credibility category)
  - `dimensions` containing `factuality`, `bias`, `intent`, and `emotion`
  - `key_findings`
  - `summary`
  - metadata like `title`, `author`, and `date`
- The trust score is computed in `inference.py` with weights:
  - factuality: 50%
  - bias: 20%
  - intent: 15%
  - emotion: 15%
- `bias` may return `Uncertain` when confidence is low.
- Frontend is intentionally simple React + Vite with vanilla CSS; avoid adding heavy UI frameworks unless requested.

## Recommended developer commands
- Backend setup
  - `python -m venv venv`
  - `venv\Scripts\activate`
  - `pip install -r requirements.txt`
- Run backend
  - `python api.py`
  - or `uvicorn api:app --reload --port 8000`
- Frontend
  - `cd frontend`
  - `npm install`
  - `npm run dev`
- Smoke test
  - `python smoke_test.py`

## When modifying the model or output
- Reference `config.py` for hyperparameters, device selection, and save/load paths.
- Keep multi-task output consistent: factuality, bias, intent, emotion, and the aggregated trust score.
- Use the `generate_report()` function if you need a stable API response format.
- Update frontend data parsing only if the report shape changes.
- Preserve the backend API contract for `/analyze` and `/analyze-text`.

## Notes for AI agents
- This repo has no existing `.github/copilot-instructions.md` or `AGENTS.md`; this file is the base agent guidance.
- Prefer linking to `README.md` and `project_prompt.md` for broader context instead of duplicating those docs.
- Focus on correctness of the model output and the stability of API responses when changing scoring logic.
