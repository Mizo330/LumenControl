###Setup base
#
#Base image can be tricky. In my opinion you should only use a few base images. Complex ones with 
#everything usually have special use cases, an in my experience they take more time to understand, 
#than building one from the ground up.
#The base iamges I suggest you to use:
#- ubuntu: https://hub.docker.com/_/ubuntu
#- nvidia/cuda: https://hub.docker.com/r/nvidia/cuda
#
#We are mostly not space constrained so a little bigger image with everything is usually better,
#than a stripped down version.


FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

#this is a small but basic utility, missing from osrf/ros. It is not trivial to know that this is
#missing when an error occurs, so I suggest installing it just to be sure.
RUN apt-get update && apt-get install -y netbase
#set shell 
SHELL ["/bin/bash", "-c"]
#set colors
ENV BUILDKIT_COLORS=run=green:warning=yellow:error=red:cancel=cyan
#start with root user
USER root

###Create new user
#
#Creating a user inside the container, so we won't work as root.
#Setting all setting all the groups and stuff.
#
###

#expect build-time argument
ARG HOST_USER_GROUP_ARG
#create group appuser with id 999
#create group hostgroup with ID from host. This is needed so appuser can manipulate the host files without sudo.
#create appuser user with id 999 with home; bash as shell; and in the appuser group
#change password of appuser to admin so that we can sudo inside the container
#add appuser to sudo, hostgroup and all default groups
#copy default bashrc and add ROS sourcing
RUN groupadd -g 999 appuser && \
    groupadd -g $HOST_USER_GROUP_ARG hostgroup && \
    useradd --create-home --shell /bin/bash -u 999 -g appuser appuser && \
    echo 'appuser:admin' | chpasswd && \
    usermod -aG sudo,hostgroup,plugdev,video,adm,cdrom,dip,dialout appuser && \
    cp /etc/skel/.bashrc /home/appuser/  

###Install the project
#
#If you install multiple project, you should follow the same 
#footprint for each:
#- dependencies
#- pre install steps
#- install
#- post install steps
#
###

#basic dependencies for everything
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive\
    apt-get install -y\
    netbase\
    git\
    build-essential\    
    wget\
    curl\
    gdb\
    lsb-release\
    sudo

#install 
#dependencies
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive\
    apt-get install -y\
    python3-pip\
    python3-distutils\
    python3-apt

USER appuser
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

#install for spot ros git clone https://github.com/bdaiinstitute/spot_ros2.git
# c056c47 this is a working commit id. Might not need it...
RUN cd /home/appuser && \
    mkdir -p lumenai/src && \
    mkdir -p lumenai/.vscode 
    

#install vscode server and extensions inside the container
#it propably won't work on computers of others because of the specific vscode version
#mostly stolen from here: https://gist.github.com/discrimy/6c21c10995f1914cf72cd8474d4501b2
#its great, because it means it is already installed into the image, so when starting a vscode instance inside the container, it will be already there.
#it will not have to download it.
#more info: https://github.com/microsoft/vscode-remote-release/issues/1718
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive\
    apt-get install -y\
    curl \
    jq
COPY  --chown=appuser:appuser ./misc/.devcontainer/ /home/appuser/lumenai/.devcontainer/
USER appuser
ARG VSCODE_COMMIT_HASH
RUN bash /home/appuser/lumenai/.devcontainer/preinstall_vscode.sh $VSCODE_COMMIT_HASH /home/appuser/lumenai/.devcontainer/devcontainer.json
COPY --chown=appuser:appuser ./src /home/appuser/lumenai/src

USER appuser
RUN python3 -m pip install tqdm && \
    python3 -m pip install imutils && \
    python3 -m pip install scikit-learn && \
    python3 -m pip install pycocotools
# RUN python3 -m pip install mediapipe
# RUN python3 -m pip install numpy==1.26.4
# RUN python3 -m pip install -U matplotlib
