# Installation

## Building the docker image
Use the command below to build a docker image:

```
# docker build -t am_combiner .
```

The python inside the docker image can be used a remote interpreter in your IDE.

If you need to use `jupyter` notebooks, you can dothe following:

```
# docker-compose -f docker-compose-development.yaml up
```

Note, that for this particular `docker-compose` file, you can also set and `env` variable called `EXCHANGE_FOLDER` that
will be automatically mounted on the service start-up. Use that folder to save your results there, or read some input
data.

## Data
The current implementation can read csv files and input, but can also read data from a mongo collection. Additionally,
a random fake entities dataset can be generated. How to do all of that, please read below.

One also has two options how to store the output results: either in a mongo collection(useful for K8S runs), or locally.

## Dev workflow
Uses `make` to run code style and unit-tests in the same way as they are run by Gitlab.
It is recommended to run these commands before committing code. Each task has its own additional requirements,
for example `make cov_test` requires the dependencies in the `requirements-dev.txt`, `make format` requires `black` and `make style` requires `flake8`.

Before committing the code, run the following:
1. `make format`

This will check if the code is properly formatted.

When you run this, it will execute a command such as `black am_combiner -l 100 --check --exclude "_nbdev|ab_utils|articles|config|date|entity_linking|profession|name_commonness"`.

If it encounters issues, re-run the command without `--check` option which will automatically format the code.

2. `make style`

This will check the style of the code, such as the line length, if there are unused imports etc.

3. `make docstyle`

This will check the docstrings.

4. `make cov_test`

This will run the unit test.

# Deploy experiments on kubernetes

Experiments can be deployed using `helm` or ArgoCD.
## Setup Experiments

These steps are required for either deployment method.

1.  Make a branch

2.  Edit the [eks-mi-playground-1-euw1-01-values.yaml](./helm/eks-mi-playground-1-euw1-01-values.yaml) file to add or change the experiments you wish to run.

    * experiment ID args are added automatically by the chart.  This will match the experimentId value provided for each experiment.

    * Remaining args are specified as a YAML array.

3.  Commit and push branch to GitLab.

## Deploy using ArgoCD

1.  When the pipeline has completed, click the deploy button.

2.  The experiment jobs will be created in ArgoCD.  Check the `entity-resolution` namespace to see these running.

## Deploy using Helm

**IMPORTANT:**  Chart has been tested with `helm3` - `helm2` is deprecated.

1.  Identify the image tag name by inspecting the output of the build stage.  For example, `feature-amc-308-define-helm-chart-e02a9335`

2.  Edit the [Chart.yaml](./helm/Chart.yaml) file.  Change the `AppVersion` field to match the image tag set above.

3.  Run the command in the repo root:

    ```sh
    helm -n entity-resolution install <experiment_name> helm/ -f helm/values.yaml -f helm/eks-mi-playground-1-euw1-01-values.yaml
    ```

4.  Jobs will now be created.


## Troubleshooting

1.  helm install reports an error / ArgoCD deploy stage fails:

    * Check the chart renders correctly using `helm template`

    * Run `helm install` and correct any reported errors.

2.  Jobs are not starting:

    * Check the kubernetes events for the entity-resolution namespace.  Jobs may not start for a number of reasons, including:  Lack of node availability, un-reported errors in the charts, or missing resources like ServiceAccounts.

    * If not automatically corrected (eg waiting for node scale up), talk to the MI SREs.

# Main script CLI

The main script for experiments running is called `__main__.py`. The script must be run form the root of the project.
The CLI description can be found in the `__main__.py` file.

## What the script does

Generally, this script has the following large blocks:

1. Load all feature extractors
2. Load all required combiners
3. Fetch input data
4. Extract features
5. Combine the data
6. Store results/statistics

## Running configuration

There are many parameters the script can be configured with. In this section we will give a couple of examples of how
to run the script.

* `--input-data-scource` allows one to configure where to fetch the input data from. It has 3 possible values:
`csv`, `mongo`, `random`. Each of these choices requires different other parameters details of which you can find
in `__main__.py`.
* `--results-storage` defines where the results will be stored. There are options: `mongo` and `local`. Again, see the
CLI description for more details how to specify where exactly the results will be stored.
* `--experiment-id` allows one to store results with different ids into different locations automatically.
* `--random-input-size` if the `random` data was selected as input, defines how many fake entities will be generated.
* `--combiners` defines a list of combiners to be run. An example combiners configuration can be found in `combiners_config.yaml`
* `--visitors` defines the features extracted for combiners to use. An example visitors configuration can be found in `combiners_config.yaml`
* `--splitter` defines an optional additional cluster breaking logic applicable only to adjacency-matrix based combiners. Configurations can be found in `combiners_config.yaml`

### Note on running splitter:

Splitter will only be applied to the combiners that uses adjacency matrix connected components approach to derive final clustering.
Fortunately, most of our combiners uses this approach. 

For example, adding `splitter=DeleteEdges_YOB_Splitter` for Sanctions will ensure that any edge joining sanctioned people 
of vastly different years of birth will be disconnected before applying connected components step.

# Continuous Integration

The CI pipeline has been optimised for speed, and performs the following tasks:

* Style checks
* Builds docker images
* Pushes helm charts
* Deploys, runs and collects the results of integration tests

The CI pipeline uses a tagged common image for the common dependencies.  To update this image, follow the process below:

1.  Add new dependencies to `requirements.txt`
2.  (Optional) Add new system requirements or dependencies to the common section of the `Dockerfile`
3.  Push these changes to a branch
4.  Release from the branch (create a new branch from the modified branch named `release.vX.Y.Z` where `X`, `Y` and `Z` are the next [semantic version][https://semver.org])
5.  Update the `BASE_IMG_TAG` variable in `.gitlab-ci.yml` to the new version selected in the previous step.
6.  Code!
7.  Raise an MR with the updated code and review.
