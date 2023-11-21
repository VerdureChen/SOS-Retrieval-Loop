ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1

# We need to create sym links for the Slurm PMI headers if we are using
# Ubuntu 18.04 because they are not installed in a standard location.
ARG UBUNTU_VERSION
RUN mkdir -p /var/run/sshd
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        autotools-dev \
        build-essential \
        ca-certificates \
        curl \
        daemontools \
        ibverbs-providers \
        libibverbs1 \
        libkrb5-dev \
        librdmacm1 \
        libssl-dev \
        libtool \
        git \
        krb5-user \
        g++ \
        cmake \
        make \
        openssh-client \
        openssh-server \
        pkg-config \
        wget \
        nfs-common \
        libnuma1 \
        libnuma-dev \
        libpmi2-0-dev \
        unattended-upgrades

# RUN unattended-upgrade \
RUN rm -rf /var/lib/apt/lists/* \
    && rm /etc/ssh/ssh_host_ecdsa_key \
    && rm /etc/ssh/ssh_host_ed25519_key \
    && rm /etc/ssh/ssh_host_rsa_key \
        && if [ "$UBUNTU_VERSION" = "ubuntu18.04" ]; then ln -s /usr/include/slurm-wlm /usr/include/slurm; fi

RUN git clone https://github.com/determined-ai/environments /tmp/determined-environments \
    && mv /tmp/determined-environments/dockerfile_scripts /tmp/det_dockerfile_scripts

ARG WITH_MPI
ARG UCX_INSTALL_DIR=/container/ucx
ARG OMPI_INSTALL_DIR=/container/ompi
RUN if [ "$WITH_MPI" = "1" ]; then /tmp/det_dockerfile_scripts/ompi.sh "$UBUNTU_VERSION" ; fi
# Make sure OMPI/UCX show up in the right paths
ARG VERBS_LIB_DIR=/usr/lib/libibverbs
ARG UCX_LIB_DIR=${UCX_INSTALL_DIR}/lib:${UCX_INSTALL_DIR}/lib64
ARG UCX_PATH_DIR=${UCX_INSTALL_DIR}/bin
ARG OMPI_LIB_DIR=${OMPI_INSTALL_DIR}/lib
ARG OMPI_PATH_DIR=${OMPI_INSTALL_DIR}/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:${WITH_MPI:+$VERBS_LIB_DIR:$UCX_LIB_DIR:$OMPI_LIB_DIR}
ENV PATH $PATH:${WITH_MPI:+$UCX_PATH_DIR:$OMPI_PATH_DIR}
# Enable running OMPI as root
ENV OMPI_ALLOW_RUN_AS_ROOT ${WITH_MPI:+1}
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM ${WITH_MPI:+1}

ARG DET_VERSION

ENV DET_PYTHON_PATH=/env/determined
ENV DET_PYTHON_EXECUTABLE=${DET_PYTHON_PATH}/bin/python3
RUN curl -o miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && /bin/bash miniconda3.sh -b -p /env/determined \
    && rm miniconda3.sh \
    && ${DET_PYTHON_PATH}/bin/conda update -n base -c defaults conda \
    && ${DET_PYTHON_PATH}/bin/conda clean -ay

# Install determined and Jupyter
RUN PYPI_SERVER=https://mirrors.aliyun.com/pypi/simple   \
    && ${DET_PYTHON_EXECUTABLE} -m pip install -i ${PYPI_SERVER} -r /tmp/det_dockerfile_scripts/notebook-requirements.txt \
    && ${DET_PYTHON_EXECUTABLE} -m pip install -i ${PYPI_SERVER} determined==${DET_VERSION}

ENV JUPYTER_CONFIG_DIR=/run/determined/jupyter/config
ENV JUPYTER_DATA_DIR=/run/determined/jupyter/data
ENV JUPYTER_RUNTIME_DIR=/run/determined/jupyter/runtime

RUN /tmp/det_dockerfile_scripts/add_det_nobody_user.sh

RUN /tmp/det_dockerfile_scripts/install_libnss_determined.sh

RUN rm -r /tmp/*

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1

# We need to create sym links for the Slurm PMI headers if we are using
# Ubuntu 18.04 because they are not installed in a standard location.
ARG UBUNTU_VERSION
RUN mkdir -p /var/run/sshd
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        autotools-dev \
        build-essential \
        ca-certificates \
        curl \
        daemontools \
        ibverbs-providers \
        libibverbs1 \
        libkrb5-dev \
        librdmacm1 \
        libssl-dev \
        libtool \
        git \
        krb5-user \
        g++ \
        cmake \
        make \
        openssh-client \
        openssh-server \
        pkg-config \
        wget \
        nfs-common \
        libnuma1 \
        libnuma-dev \
        libpmi2-0-dev \
        unattended-upgrades

# RUN unattended-upgrade \
RUN rm -rf /var/lib/apt/lists/* \
    && rm /etc/ssh/ssh_host_ecdsa_key \
    && rm /etc/ssh/ssh_host_ed25519_key \
    && rm /etc/ssh/ssh_host_rsa_key \
        && if [ "$UBUNTU_VERSION" = "ubuntu18.04" ]; then ln -s /usr/include/slurm-wlm /usr/include/slurm; fi

RUN git clone https://github.com/determined-ai/environments /tmp/determined-environments \
    && mv /tmp/determined-environments/dockerfile_scripts /tmp/det_dockerfile_scripts

ARG WITH_MPI
ARG UCX_INSTALL_DIR=/container/ucx
ARG OMPI_INSTALL_DIR=/container/ompi
RUN if [ "$WITH_MPI" = "1" ]; then /tmp/det_dockerfile_scripts/ompi.sh "$UBUNTU_VERSION" ; fi
# Make sure OMPI/UCX show up in the right paths
ARG VERBS_LIB_DIR=/usr/lib/libibverbs
ARG UCX_LIB_DIR=${UCX_INSTALL_DIR}/lib:${UCX_INSTALL_DIR}/lib64
ARG UCX_PATH_DIR=${UCX_INSTALL_DIR}/bin
ARG OMPI_LIB_DIR=${OMPI_INSTALL_DIR}/lib
ARG OMPI_PATH_DIR=${OMPI_INSTALL_DIR}/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:${WITH_MPI:+$VERBS_LIB_DIR:$UCX_LIB_DIR:$OMPI_LIB_DIR}
ENV PATH $PATH:${WITH_MPI:+$UCX_PATH_DIR:$OMPI_PATH_DIR}
# Enable running OMPI as root
ENV OMPI_ALLOW_RUN_AS_ROOT ${WITH_MPI:+1}
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM ${WITH_MPI:+1}

ARG DET_VERSION

ENV DET_PYTHON_PATH=/env/determined
ENV DET_PYTHON_EXECUTABLE=${DET_PYTHON_PATH}/bin/python3
RUN curl -o miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && /bin/bash miniconda3.sh -b -p /env/determined \
    && rm miniconda3.sh \
    && ${DET_PYTHON_PATH}/bin/conda update -n base -c defaults conda \
    && ${DET_PYTHON_PATH}/bin/conda clean -ay

# Install determined and Jupyter
RUN PYPI_SERVER=https://mirrors.aliyun.com/pypi/simple   \
    && ${DET_PYTHON_EXECUTABLE} -m pip install -i ${PYPI_SERVER} -r /tmp/det_dockerfile_scripts/notebook-requirements.txt \
    && ${DET_PYTHON_EXECUTABLE} -m pip install -i ${PYPI_SERVER} determined==${DET_VERSION}

ENV JUPYTER_CONFIG_DIR=/run/determined/jupyter/config
ENV JUPYTER_DATA_DIR=/run/determined/jupyter/data
ENV JUPYTER_RUNTIME_DIR=/run/determined/jupyter/runtime

RUN /tmp/det_dockerfile_scripts/add_det_nobody_user.sh

RUN /tmp/det_dockerfile_scripts/install_libnss_determined.sh

RUN rm -r /tmp/*

ENV DET_SKIP_PIP_INSTALL=true
# 使用 /env/determined/bin/jupyter 替换默认 Python 入口的 jupyter
RUN ln -sf /env/determined/bin/jupyter $(which jupyter)

