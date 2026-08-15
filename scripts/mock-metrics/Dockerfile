FROM docker.m.daocloud.io/library/python:3.12-alpine
WORKDIR /app
COPY server.py .
EXPOSE 9108
CMD ["python", "-u", "server.py", "--host", "0.0.0.0", "--port", "9108"]
