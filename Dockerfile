FROM python:3.11
WORKDIR /usr/src/query-pdfs
COPY . .
RUN chmod +x setup.sh && ./setup.sh
ENV PATH="/usr/src/query-pdfs/venv/bin:${PATH}"
ARG GOOGLE_API_KEY
CMD ["streamlit", "run", "app.py"]