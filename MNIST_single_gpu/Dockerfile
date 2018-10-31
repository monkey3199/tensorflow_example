FROM docker.io/nvidia/cuda:8.0-cudnn5-devel-centos7
RUN yum -y install epel-release
RUN yum -y install python-pip
RUN pip install tensorflow_gpu==1.0.0
RUN mkdir -p /opt/mnist
COPY mnist_cnn.py /opt/mnist/
ENTRYPOINT ["python", "/opt/mnist/mnist_cnn.py"]
