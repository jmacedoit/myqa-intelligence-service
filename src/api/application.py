
from flask import Flask
from .controllers.knowledge_bases import knowledge_bases_blueprint
from .controllers.answers import answers_blueprint

app = Flask(__name__)

app.register_blueprint(knowledge_bases_blueprint)
app.register_blueprint(answers_blueprint)
