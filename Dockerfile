FROM ubuntu

FROM python

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt
EXPOSE 5000
WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

CMD  python /app/app_flask.py

