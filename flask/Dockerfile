FROM ubuntu:latest
RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /covid/requirements.txt

WORKDIR /covid

RUN pip install -r requirements.txt

COPY . /covid

ENTRYPOINT [ "python" ]

CMD [ "covid-routing.py" ]
