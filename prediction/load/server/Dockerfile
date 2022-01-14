FROM python:3.7.3

COPY . ./load_server

WORKDIR /load_server

RUN apt-get update && apt-get install -y default-jre

RUN pip install --upgrade pip 

RUN pip install -r server_requirements.txt

EXPOSE 8876

CMD ["python", "load_server.py"]
