# The graph visualization web server
The graph visualization server is used to render the graph data in a 3d format which is more intuitive in terms of understanding the various mentions and connections among them w.r.t the extrated entities

## Steps to run the flask app
1. Start the flask serve using the command app.py from your docker or virtual env
2. Open 0.0.0.0:5000/ENTITYNAME/FOLDER_PATH to render the graph viz for a particular entity and its associated mentions

*   ENTITYNAME: The entity of choice for which we want to get the graph
*   FOLDER_PATH: The path where the graph data is stored as pickles. The GraphDataVisitor generates the graph data when its run using __main__.py file

A working request to get a 3d Visualiation of connected mentions for an entity will be

                **0.0.0.0:5000/John Roberts/home/tdhar/src/am_combiner/exchange/2/graph_data**