# Training a model:

## Train a model in jupyter notebook in the following steps:

#### Step 1: 
Introduce training features:

```
from am_combiner.combiners.ml_training import GCNModelTraining
from am_combiner.features.article import Features

training = GCNModelTraining(
    node_features = Features.TFIDF_FULL_TEXT_12000,
    edge_features = [
        Features.ORG_CLEAN,
        Features.PERSON_CLEAN,
        Features.GPE_CLEAN,
        Features.LOC,
        Features.DOMAIN,
        Features.AM_CATEGORY, 
    ],
    )
```

#### Step 2: 
Load training data with custom data aggregation function. 
Here, both train and test sets require both dataframe with ClusterID column and dictionary with article objects.

For Homogeneous graph, good start would be ```articles_to_homogeneous_graph()``` from ```am_combiner.features.nn.common```
For Heterogeneous case, consider starting with using ```articles_to_hetero_graph()```

```
training.add_train_test_data(    
    input_test_entities, 
    input_train_entities, 
    articles_dict_test, 
    articles_dict_train,
    articles_to_homogeneous_graph,
    num_workers=2,
)
```

#### Step 3:
Train the model.
if the model architecture is already present in am_combiner codebase - feel free to import it. Otherwise, it can be introduced in the notebook.

```
from am_combiner.combiners.ml import GCN
training.net = GCN(8000, 128)
training.train_model()
```

#### Step 4:
If model performance is satisfactory, save model together with json file storing initialisation parameters.
```
model_path, config_path = training.save_best_model(model_name="my_favourite_model_name")
```
