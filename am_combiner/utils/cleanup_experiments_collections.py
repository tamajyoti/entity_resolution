import click
from pymongo import MongoClient


@click.command(context_settings={"show_default": True})
@click.option(
    "--mongo-uri",
    is_eager=True,
    help="Mongo URI for whatever reasons you may need it",
    default="mongodb://"
    "mongodb-mf-person-0-0.mi-playground-1"
    "/?replicaSet=mf-person-0&readPreference=primary&appname=MongoDB%20Compass&ssl=false",
    expose_value=True,
)
@click.option(
    "--mongo-database",
    is_eager=True,
    help="The name of input database to be used as a combiner input",
    default="er-k8s-cluster-results",
)
@click.option(
    "--mongo-collection",
    is_eager=True,
    help="The name of input collection to be used as a combiner input",
    required=True,
    multiple=True,
)
def main(mongo_uri, mongo_database, mongo_collection):
    """Cleanup collections used in automated experiments."""
    db = MongoClient(mongo_uri)[mongo_database]
    for mc in mongo_collection:
        db[mc].drop()


if __name__ == "__main__":
    main()
