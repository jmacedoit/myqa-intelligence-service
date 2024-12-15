
from .controllers.knowledge_bases import knowledge_bases_blueprint
from .controllers.answers import answers_blueprint
from .controllers.chunks import chunks_blueprint
from api.server_application import app


app.register_blueprint(knowledge_bases_blueprint)
app.register_blueprint(chunks_blueprint)
app.register_blueprint(answers_blueprint)
