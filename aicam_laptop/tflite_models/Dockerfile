# This is a linux image with the edgetpu compiler installed. It is used to compile the tflite models for the coral edge tpu.
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    ca-certificates \
    && echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    > /etc/apt/sources.list.d/coral-edgetpu.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update \
    && apt-get install -y edgetpu-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work