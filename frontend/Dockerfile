FROM python:3.12-slim

WORKDIR /app


COPY requirements-frontend.txt .
RUN pip install --no-cache-dir -r requirements-frontend.txt


COPY frontend/app.py ./frontend/app.py

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
