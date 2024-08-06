FROM python:3.12

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./models/. /code/models/.

COPY ./backend_app /code/backend_app

EXPOSE 8000

CMD ["uvicorn", "backend_app.app:app", "--host","0.0.0.0","--port","8000"]