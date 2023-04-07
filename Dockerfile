# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the dependencies file to the working directory
COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip install --no-cache-dir poetry

# Install dependencies using Poetry
RUN poetry install --no-root --no-dev

# Copy the rest of the application code to the working directory
COPY . .

# Set the command to run the application
CMD ["poetry", "run", "python", "src/main.py"]