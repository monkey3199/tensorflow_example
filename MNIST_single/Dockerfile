FROM  tensorflow/tensorflow:1.8.0
# RUN pip install matplotlib
RUN mkdir -p /opt/mnist
# COPY mnist.py /opt/mnist/
COPY mnist_cnn.py /opt/mnist/
# ENTRYPOINT ["python", "/opt/mnist/mnist.py"]
ENTRYPOINT ["python", "/opt/mnist/mnist_cnn.py"]
