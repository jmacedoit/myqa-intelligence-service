# myqa-intelligence-service
This service provides myqa intelligence layer

## Requirements
Docker or:

  * Python >= 3.9.x (You can use `pyenv` to manage the python version)
  * Poetry

## Setup
An easy way to set the service running is using the dockerfile available.

But if you are contributing to the project, simply run:

```
poetry install
```

## Run
In order to execute the project follow [configuration instructions](##Configuration) to set OpenAI key and then simply execute the following command:

```
poetry run python src/main.py
```

## Configuration
Settings can be changed in `settings.json`. Secrets like OpenAI api key can be simply set by creating a file named `.secrets.json` in the `src` folder and then filling the necessary settings following the same structure as `settings.json`.

## Docker
A Dockerfile is supplied with this project. To build the image run:
```
docker build -t myqa-intelligence-service .
```

Then run the image. 
```
docker run -it -e VAR1=/mnt/output -p 7000:700 nmyqa-intelligence-service
```