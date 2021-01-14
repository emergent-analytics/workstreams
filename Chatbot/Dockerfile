FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
RUN pip install --no-cache-dir pandas sqlalchemy ibm-db-sa
EXPOSE 80
COPY ./app/ /app
