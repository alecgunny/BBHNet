# BBHNet
Detecting binary blackhole mergers from gravitational wave strain timeseries data using neural networks, with a focus on
- *Efficiency* - making effective use of accelerated hardware like GPUs in order to minimize time-to-solution.
- *Scale* - validating hypotheses on large volumes of data to obtain high-confidence estimates of model performance
- *Flexibility* - modularizing functionality to expose various levels of abstraction and make implementing new ideas simple
- *Physics first* - taking advantage of the rich priors available in GW physics to build robust models and evaluate them accoring to meaningful metrics

BBHNet represents a _framework_ for optimizing neural networks for detection of CBC events from time-domain strain, rather than any particular network architecture.

## Run an experiment
> **_NOTE:_** right now, BBHNet can only be run by LIGO members
> **_NOTE:_** Running BBHNet out-of-the-box requires access to an enterprise-grade GPU (e.g. P100, V100, T4, A[30,40,100], etc.). There are several nodes on the LIGO Data Grid which meet these requirements.

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

```
mkdir -p ~/cilogon_cert ~/bbhnet/data ~/bbhnet/results

### With singularity
> **_NOTE:_** Runing
The easiest way to get started with BBHNet is to run the [`sandbox`](./projects/sandbox) experiment using our pre-built container. If you're on a GPU-enabled node LIGO Data Grid (LDG), and you set up the directories as outlined above, just run

```
singularity exec --nv \
    --bind ~/bbhnet/results:/opt/bbhnet/results \
    --bind ~/bbhnet/data:/opt/bbhnet/data \
    --bind ~/.kerberos:/root/.kerberos \
    --bind ~/cilogon_cert:/root/cilogon_cert \
    --bind /cvmfs:/cvmfs \
    /cvmfs/singularity.opensciencegrid.org/ml4gw/bbhnet \
    pinto -p /opt/bbhnet/src/projects/sandbox run
```
This will download background and glitch datasets and generate a dataset of raw gravitational waveforms, train a model on this data, perform inference using the trained model on a dataset of timeshifted data with injections, and
serve up an application for visualizing and analyzing those outputs at `localhost:5005`.

### With `pinto` (aka Conda/Poetry, recommended for development)
You can also run BBHNet on bare metal by cloning this repository

```console
git clone git@github.com:ML4GW/bbhnet.git
```

and installing the [Pinto environment management tool](https://github.com/ml4gw/pinto#installation). This method is recommended if you plan on doing any development on BBHNet.
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


## Structure
The code here is structured like a [monorepo](https://medium.com/opendoor-labs/our-python-monorepo-d34028f2b6fa), with applications siloed off into isolated environments to keep dependencies lightweight, but built on top of a shared set of libraries to keep the code modular and consistent.
