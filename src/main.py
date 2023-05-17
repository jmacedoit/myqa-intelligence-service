
from api.server_application import socketio
from api.application import app
from config import settings;
from pymilvus import connections;

connections.connect(host=settings.milvus.host, port=settings.milvus.port)

socketio.run(app, port=settings.server.port)

