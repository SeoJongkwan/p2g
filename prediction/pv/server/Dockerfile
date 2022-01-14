FROM python:3.7.3

COPY . ./pv_server

WORKDIR /pv_server

RUN apt-get update && apt-get install -y default-jre

RUN pip install --upgrade pip 

RUN pip install -r server_requirements.txt

EXPOSE 8856

CMD ["python", "pv_server.py"]
