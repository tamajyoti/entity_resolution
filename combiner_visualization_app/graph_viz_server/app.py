from flask import Flask
from flask_cors import CORS
from config import Config
import os
from graph_viz import entity_graph_blueprint


# Initialize and Register the flask blueprint
def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    app.register_blueprint(entity_graph_blueprint)
    CORS(app)

    return app.run(debug=True,
                   host=os.getenv('IP', '0.0.0.0'))

# Run the app
if __name__ == '__main__':
    create_app()
