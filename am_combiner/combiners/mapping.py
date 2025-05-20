from am_combiner.combiners.graph_based import ConnectedComponentsCombiner
from am_combiner.combiners.incremental_clustering import (
    PairwiseIncrementalCombiner,
    CentroidIncrementalCombiner,
)

from am_combiner.combiners.clusterer import LibTFIDFCosineSimilarityClusterer
from am_combiner.combiners.ml import (
    GCNCombiner,
    GCNHeteroCombiner,
    SklearnClassificationModelBasedCombiner,
    GCNCombinerWithLinearCombination,
)
from am_combiner.combiners.simple import CurrentProductionCombiner
from am_combiner.combiners.annotation import AnnotationsCombiner
from am_combiner.combiners.fastRP import FastRPCosineSim
from am_combiner.combiners.tfidf import (
    TFIDFKMeansCombiner,
    TFIDFCosineSimilarityCombiner,
    TFIDFAndFeaturesCosineSimilarityCombiner,
    TFIDFAndGraphCosineSimilarityCombiner,
    TFIDFFeatrGraphCosineSimilarityCombiner,
)

COMBINER_CLASS_MAPPING = {
    "ConnectedComponentsCombiner": ConnectedComponentsCombiner,
    "CurrentProductionCombiner": CurrentProductionCombiner,
    "TFIDFKMeansCombiner": TFIDFKMeansCombiner,
    "TFIDFCosineSimilarityCombiner": TFIDFCosineSimilarityCombiner,
    "SklearnClassificationModelBasedCombiner": SklearnClassificationModelBasedCombiner,
    "PairwiseIncrementalCombiner": PairwiseIncrementalCombiner,
    "CentroidIncrementalCombiner": CentroidIncrementalCombiner,
    "TFIDFAndFeaturesCosineSimilarityCombiner": TFIDFAndFeaturesCosineSimilarityCombiner,
    "LibTFIDFCosineSimilarityClusterer": LibTFIDFCosineSimilarityClusterer,
    "TFIDFAndGraphCosineSimilarityCombiner": TFIDFAndGraphCosineSimilarityCombiner,
    "TFIDFFeatrGraphCosineSimilarityCombiner": TFIDFFeatrGraphCosineSimilarityCombiner,
    "AnnotationsCombiner": AnnotationsCombiner,
    "GCNCombiner": GCNCombiner,
    "GCNHeteroCombiner": GCNHeteroCombiner,
    "GCNCombinerWithLinearCombination": GCNCombinerWithLinearCombination,
    "FastRPCosineSim": FastRPCosineSim,
}
