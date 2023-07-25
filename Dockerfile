# Set the python version for the parent image
ARG PYTHON_VERSION=""

# Use an official Python runtime as a parent image
FROM python:${PYTHON_VERSION}-buster

# Redefine the ARG after the FROM statement
ARG POETRY_VERSION=""

# Set the working directory in the container to /workspace
WORKDIR /workspace

# Update and install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --no-cache-dir -U pip setuptools wheel

# Install Poetry
RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

# Configure Poetry
RUN poetry config virtualenvs.create true \
    && poetry config virtualenvs.in-project true
