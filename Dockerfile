FROM python:3.10

WORKDIR /code

COPY requirements.txt ./code/requirements.txt

RUN pip install -r ./code/requirements.txt

COPY ./app ./code/app

RUN $HOME/sbin/start-connect-server.sh --packages org.apache.spark:spark-connect_2.12:$SPARK_VERSION