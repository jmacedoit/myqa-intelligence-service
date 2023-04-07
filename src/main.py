
from api.application import app
from config import settings;
from pymilvus import connections;

connections.connect(host=settings.milvus.host, port=settings.milvus.port)

app.run(port=settings.server.port)
