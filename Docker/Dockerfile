## Start with Docker pytorch base
FROM nvidia/cuda:10.1-base-ubuntu16.04
ARG PYTHON_VERSION=3.6.9

# Install custom libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
         curl git && \
         rm -rf /var/lib/apt/lists/*



# Install Miniconda and Python 3.6
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p /opt/conda \
 && rm ~/miniconda.sh \
 && /opt/conda/bin/conda install -y python==$PYTHON_VERSION \
 && /opt/conda/bin/conda install -y -c pytorch cudatoolkit=10.1 "pytorch=1.4.0=py3.6_cuda10.1.243_cudnn7.6.3_0" "torchvision=0.5.0=py36_cu101" \
 && /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH


# install dependencies from python packages
RUN pip install --upgrade pip \
&& pip3 install nibabel==2.5.1 numpy==1.17.4 scipy==1.3.1 scikit-image==0.15.0 pandas==0.21.0 matplotlib==3.0.3

## Clone FatSegnet Code from repository
RUN mkdir /FatSegNet
RUN git clone https://github.com/reuter-lab/FatSegNet_pytorch.git /FatSegNet

WORKDIR "/FatSegNet/fatsegnet"

ENTRYPOINT ["python3","./run_fatsegnet.py"]
#ENTRYPOINT ["bash"]