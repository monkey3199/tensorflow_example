FROM cuda:8.0-cudnn5-runtime
RUN mkdir -p /opt/mnist
COPY mnist_cnn.py /opt/mnist/
ENTRYPOINT ["python", "/opt/mnist/mnist_cnn.py"]
