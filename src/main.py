
import json
from api.server_application import socketio
from api.application import app
from config import settings, settings_without_secrets;
from pymilvus import connections;
from logger import logger


logger.info("Settings *WITHOUT* secrets:")
logger.info(json.dumps(settings_without_secrets.as_dict(), indent=4))

connections.connect(host=settings.milvus.host, port=settings.milvus.port)

socketio.run(app, port=settings.server.port, host='0.0.0.0')

