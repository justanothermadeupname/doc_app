#!/usr/bin/env bash
python embed_dbqs.py          # rebuild vector index
exec uvicorn main:app --host 0.0.0.0 --port 8000
