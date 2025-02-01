gunicorn -w 4 -k uvicorn.workers.UvicornWorker plantdiseaseserver:app
uvicorn plantdiseaseserver:app --host 172.22.158.78 --port 8000 --reload