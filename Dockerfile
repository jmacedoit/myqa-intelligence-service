# Use an official Python runtime as a parent image
FROM --platform=linux/amd64 python:3.10.11-buster

RUN apt-get update
RUN apt-get install -y build-essential

# Set the working directory to /app
WORKDIR /app

# Copy the dependencies file to the working directory
COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root --no-dev

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application code to the working directory
COPY . .

EXPOSE 7000

# Set the command to run the application
CMD ["poetry", "run", "python", "src/main.py"]