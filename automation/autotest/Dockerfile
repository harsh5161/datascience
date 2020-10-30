FROM python:3.8.0
VOLUME /testlogs
RUN apt-get update \
&& apt-get install gcc -y \
&& apt-get clean
COPY requirements.txt /app/requirements.txt
WORKDIR app
RUN pip install --upgrade pip
RUN  pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY . /app
CMD ["python", "./testtk.py"]