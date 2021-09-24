FROM ubuntu:18.04
COPY mnist-example /exp/mnist-example
COPY requirements.txt /exp/requirements.txt
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
RUN mkdir /exp/models
WORKDIR /exp
CMD ["python3", "./mnist-example/plot_graphs.py"]
