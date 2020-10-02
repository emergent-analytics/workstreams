FROM python:3.8-slim
RUN pip install --no-cache-dir streamlit pandas numpy plotly scipy sqlalchemy Pillow
WORKDIR /app
COPY A_UK.csv /app
COPY GVA_UK.csv /app/GVA_uk.csv
COPY SimEngine_v5.py /app
EXPOSE 8501/tcp
ENTRYPOINT streamlit run SimEngine_v5.py
