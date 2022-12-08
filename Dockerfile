FROM ghcr.io/ml4gw/pinto:main

# establish default environment variable values
# needed to run experiments and fetch data
ENV BASE_DIR=$BBHNET/results \
    DATA_DIR=$BBHNET/data \
    LIGO_USERNAME=albert.einstein \
    KRB5_KTNAME=/root/.kerberos/ligo.org.keytab \
    X509_USER_PROXY=/root/cilogon_cert/CERT_KEY.pem \
    DEBIAN_FRONTEND=noninteractive

# estabilsh directories and ports to be mapped into from host
VOLUME /cvmfs /root/.kerberos /root/cilogon_cert $BASE_DIR $DATA_DIR
EXPOSE 5005

# install singularity so that we can run triton
# from inside container
RUN set +x \
        \
        # install apt dependencies
        apt-get update && apt-get install -y --no-install-recommends \
            build-essential \
            libssl-dev \
            uuid-dev \
            libgpgme11-dev \
            squashfs-tools \
            libseccomp-dev \
            pkg-config \
            wget \
        \
        # install Go
        && export VERSION=1.11 OS=linux ARCH=amd64 \
        \
        && wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz \
        \
        && tar -C /usr/local -xzvf go$VERSION.$OS-$ARCH.tar.gz \
        \
        && rm go$VERSION.$OS-$ARCH.tar.gz \
        \
        # install NeuroDebian source repo
        && wget -O- \
            http://neuro.debian.net/lists/bullseye.us-ca.libre | \
            tee /etc/apt/sources.list.d/neurodebian.sources.list \
        \
        && apt-key adv --recv-keys --keyserver hkps://keyserver.ubuntu.com 0xA5D32F012649A5A9 \
        \
        && apt-get update
        \
        # install singularity
        && apt-get install -y --no-install-recommends singularity-container \
        \
        && singularity --version \
        \
        # cleanup
        && rm -rf /var/lib/apt/lists/*

# move the repo into the container and
# set it as the working directory
ENV BBHNET=/opt/bbhnet
COPY . $BBHNET/src
WORKDIR $BBHNET/src

# install all the project environments
RUN set +x \
        \
        && cd projects/sandbox \
        \
        && for d in $(ls -d */); do pinto -p $d build; done \
        \
        & rm -rf ~/.cache/pypoetry/*
