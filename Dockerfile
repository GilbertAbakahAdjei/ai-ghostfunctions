FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install
RUN poetry run pytest