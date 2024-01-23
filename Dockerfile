FROM python:3.10
WORKDIR /usr/src/query-pdfs
RUN source setup.sh
COPY . .
CMD ["streamlit", "run", "app.py"]