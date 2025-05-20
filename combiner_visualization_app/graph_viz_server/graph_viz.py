from flask import Blueprint, render_template
from pathlib import Path
import pandas as pd

# The blueprint for the entity graph
entity_graph_blueprint = Blueprint('entity_graph', __name__, template_folder='templates')


@entity_graph_blueprint.route("/<entity_name>/<path:folder_path>", methods=["GET"])
def entity_graph(entity_name, folder_path):
    # create the folder path. A / is added as an hack to handle flask routing at the start of folder path
    folder_path = "/"/Path(folder_path)/f"{entity_name}.pickle"
    # get the graph data
    graph_data = pd.read_pickle(folder_path)
    return render_template("graph_viz.html", name=graph_data.name[0], total_articles=graph_data.articles[0],
                           output_graph=graph_data.graph_data[0])
