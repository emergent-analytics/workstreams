FROM python:3.8-slim
RUN pip install --no-cache-dir streamlit pandas numpy plotly scipy sqlalchemy Pillow
WORKDIR /app
COPY *.csv /app/
COPY *.py /app/
EXPOSE 8501/tcp
ENTRYPOINT streamlit run SimEngine.py
