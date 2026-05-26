## Extraction Test Suite

### What this is
`sample_complaints/` contains example complaint JSON payloads.

`run_extraction_suite.ps1` posts each sample to the backend `POST /extract` and writes results to `tests/results/<timestamp>/`.

### Run
1. Start backend:

```powershell
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

2. Run suite from repo root:

```powershell
.\tests\run_extraction_suite.ps1
```

Optional (custom API base):

```powershell
.\tests\run_extraction_suite.ps1 -ApiBase "http://127.0.0.1:8000"
```

### Add new samples
Drop more `*.json` files into `tests/sample_complaints/` with shape:

```json
{ "text": "raw complaint text here" }
```

