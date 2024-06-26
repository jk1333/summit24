FROM python:3.10

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r  requirements.txt 

COPY Main.py .

CMD streamlit run Main.py --server.port=8080