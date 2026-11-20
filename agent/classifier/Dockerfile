FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
ENV PORT=8080
# one worker: the assets (~50 MB on disk, ~300 MB in memory) are loaded once per instance
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 120 app:app
