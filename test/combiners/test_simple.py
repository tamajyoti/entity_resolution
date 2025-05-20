from am_combiner.combiners.simple import CurrentProductionCombiner
from am_combiner.features.article import Article


def test_production_clustering(article):

    articles = [
        Article("John Smith", "Some first text", "some.first.url"),
        Article("John Smith", "Some more text", "some.second.url"),
    ]
    combiner = CurrentProductionCombiner()
    combined_df = combiner.combine_entities(articles)

    assert list(combined_df.ClusterID.unique()) == [0]
    assert combined_df.unique_id.tolist() == ["some.first.url", "some.second.url"]
    assert list(combined_df.blocking_field.unique()) == ["John Smith"]
