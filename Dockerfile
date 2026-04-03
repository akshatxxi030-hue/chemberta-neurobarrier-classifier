FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

# install CPU torch
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# install rest
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]