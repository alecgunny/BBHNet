# BBHNet
Detecting binary blackhole mergers from gravitational wave strain timeseries data using neural networks, with an emphasis
- **Efficiency** - making effective use of accelerated hardware like GPUs in order to minimize time-to-solution.
- **Scale** - validating hypotheses on large volumes of data to obtain high-confidence estimates of model performance
- **Flexibility** - modularizing functionality to expose various levels of abstraction and make implementing new ideas simple
- **Physics first** - taking advantage of the rich priors available in GW physics to build robust models and evaluate them accoring to meaningful metrics

BBHNet represents a _framework_ for optimizing neural networks for detection of CBC events from time-domain strain, rather than any particular network architecture.

## Quickstart
> **_NOTE:_** right now, BBHNet can only be run by LIGO members

> **_NOTE:_** Running BBHNet out-of-the-box requires access to an enterprise-grade GPU (e.g. P100, V100, T4, A[30,40,100], etc.). There are several nodes on the LIGO Data Grid which meet these requirements.

### Setting up your environment
In order to access the LIGO data services required to run BBHNet, start by following the instructions [here](https://computing.docs.ligo.org/guide/auth/kerberos/#usage) to set up a kerberos keytab for passwordless authentication to LIGO data services

```console
$ ktutil
ktutil:  addent -password -p albert.einstein@LIGO.ORG -k 1 -e aes256-cts-hmac-sha1-96
Password for albert.einstein@LIGO.ORG:
ktutil:  wkt ligo.org.keytab
ktutil:  quit
```
with `albert.einstein` replaced with your LIGO username. Move this keytab file to `~/.kerberos`

```console
mkdir ~/.kerberos
mv ligo.org.keytab ~/.kerberos
```
You'll also want to create directories for storing X509 credentials, input data, and BBHNet outputs.

```console
mkdir -p ~/cilogon_cert ~/bbhnet/data ~/bbhnet/results
```

### Run with singularity
The easiest way to get started with BBHNet is to run the [`sandbox`](./projects/sandbox) experiment using our pre-built container. If you're on a GPU-enabled node LIGO Data Grid (LDG), and you set up the directories as outlined above, start by defining a couple environment variables

```console
# BASE_DIR is where we'll write all logs, training checkpoints,
# and inference/analysis outputs. This should be unique to
# each experiment you run
BASE_DIR=~/bbhnet/results/my-first-run

# DATA_DIR is where we'll write all training/testing
# input data, which can be reused between experiment
# runs. Just be sure to delete existing data or use
# a new directory if a new experiment changes anything
# about how data is generated, because BBHNet by default
# will opt to use cached data if it exists.
DATA_DIR=~/bbhnet/data
```

then you can just run

```console
apptainer exec --nv --writable-tmpfs \
    `# map in our credentials` \
    --bind ~/.kerberos:/root/.kerberos \
    --bind ~/cilogon_cert:/root/cilogon_cert \
    `# map in some helpful directories from LDG` \
    --bind /hdfs:/hdfs \
    --bind /cvmfs:/cvmfs \
    --bind /etc/condor:/etc/condor \
    `# map in our results and data directories` \
    --bind $BASE_DIR:/opt/bbhnet/results \
    --bind $DATA_DIR:/opt/bbhnet/data \
    /cvmfs/singularity.opensciencegrid.org/ml4gw/bbhnet \
        pinto -p /opt/bbhnet/src/projects/sandbox run
```

This will
- Download background and glitch datasets and generate a dataset of raw gravitational waveforms
- Train a 1D ResNet architecture on this data
- Accelerate the trained model using TensorRT and export it for as-a-service inference
- Serve up this model with Triton Inference Server via Singularity, and use it to run inference on a dataset of timeshifted background and waveform-injected strain data
- Use these predictions to generate background and foreground event distributions
- Serve up an application for visualizing and analyzing those distributions at `localhost:5005`.


### Develop with Singularity
If you want to play with BBHNet's code and run novel experiments that _don't_ require you to update the Python environments the code runs in, you can just map your local changes into the container at run time:

```console
# assumes you're running from the root of this repo
apptainer exec --nv --writable-tmpfs \
    --bind ~/.kerberos:/root/.kerberos \
    --bind ~/cilogon_cert:/root/cilogon_cert \
    --bind /hdfs:/hdfs \
    --bind /cvmfs:/cvmfs \
    --bind /etc/condor:/etc/condor \
    --bind $BASE_DIR:/opt/bbhnet/results \
    --bind $DATA_DIR:/opt/bbhnet/data \
    `# bind our local volume into the container` \
    --bind $PWD:/opt/bbhnet/src \
    /cvmfs/singularity.opensciencegrid.org/ml4gw/bbhnet \
        pinto -p /opt/bbhnet/src/projects/sandbox run
```

## Experiment overview
### Binary blackhole detection with deep learning
Overview of problem and how we're trying to solve it

### Code Structure
The code here is structured like a [monorepo](https://medium.com/opendoor-labs/our-python-monorepo-d34028f2b6fa), with applications siloed off into isolated environments to keep dependencies lightweight, but built on top of a shared set of libraries to keep the code modular and consistent.

## Development instructions
Start by cloning this repo

```console
git clone git@github.com:ML4GW/bbhnet.git
```

and then follow the [environment setup instructions](#setting-up-your-environment). Once your environment is ready, the simplest way to manage the Python environments required by the various projects within BBHNet is by using the [Pinto environment management tool](https://github.com/ml4gw/pinto#installation).

`pinto` is a simple wrapper around Poetry and Conda to combine their functionality in a way that obscures some of the more complicated ways they interact. If you ever find your environments breaking, you can always delete them and rebuild them using the appropriate combination of `poetry` and `conda` commands to get a more detailed analysis of where things are breaking.

Once `pinto` is installed (which you can confirm with `pinto --version`), you can run the sandbox experiment by setting all the appropriate environment variables

```console
# these I would recommend setting globally in your ~/.bashrc
LIGO_USERNAME=<insert your ligo username here>
KRB5_KTNAME=~/.kerberos/ligo.org.keytab
X509_USER_PROXY=~/cilogon_cert/CERT_KEY.pem

# these I would recommend setting locally in
# a file projects/sandbox/.env
BASE_DIR=~/bbhnet/results/my-first-run
DATA_DIR=~/bbhnet/data
```

and running `pinto -p projects/sandbox run` from this directory.
