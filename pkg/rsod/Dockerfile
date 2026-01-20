FROM docker.io/library/localhost:ubuntu-24.04-mingw64

RUN apt-get update && \
    apt-get install -y g++ libboost-all-dev

RUN apt-get install -y build-essential libc6-dev gcc-multilib

ENV CPLUS_INCLUDE_PATH=/usr/include:$CPLUS_INCLUDE_PATH
ENV LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

WORKDIR /workspace