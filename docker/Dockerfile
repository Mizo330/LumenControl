###Setup base
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

#this is a small but basic utility, missing from osrf/ros. It is not trivial to know that this is
#missing when an error occurs, so I suggest installing it just to be sure.
RUN apt-get update && apt-get install -y netbase
#set shell 
SHELL ["/bin/bash", "-c"]
#set colors
ENV BUILDKIT_COLORS=run=green:warning=yellow:error=red:cancel=cyan

#set noninteractive mode
ENV DEBIAN_FRONTEND=noninteractive

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
#create group appuser with id 1001
#add appuser to the group with the same id as the host user
#this is important, because it allows to mount the host user home directory into the container
#this way, the container will have the same permissions as the host user
#create appuser user with id 1001 with home; bash as shell; and in the appuser group
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
    apt-get install -y\
    netbase\
    git\
    build-essential\    
    wget\
    curl\
    gdb\
    lsb-release\
    sudo\
    portaudio19-dev\
    gdb \
    python3-pip\
    python3-apt\    
    clang-tidy

#install 
#dependencies
USER root
RUN apt-get update && \
    apt-get install -y\
    python3-pip\
    python3-distutils\
    python3-apt \
    libasound-dev libportaudio2 libportaudiocpp0 portaudio19-dev \
    pulseaudio\
    sox libsox-fmt-all\
    alsa-utils

#set dynamic distro arg name with default value
ARG ROS_DISTRO=humble

#install ROS2 https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html
USER root
RUN rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive\
    apt-get install -y ros-$ROS_DISTRO-desktop \
    ros-dev-tools

#ros sourcing
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /home/appuser/.bashrc

USER appuser
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

USER appuser
RUN python3 -m pip install tqdm scikit-learn&& \
    #this is the Python Dmx control interface.
    python3 -m pip install -U PyDMXControl
    #Pytorch extension for better performance
    #python3 -m pip install natten==0.17.1+torch240cu118 -f https://shi-labs.com/natten/wheels &&\
    #all in one music structure analizer (what a bad module name..)
    #python3 -m pip install git+https://github.com/CPJKU/madmom  allin1

USER appuser
RUN python3 -m pip install -U\
    Pyaudio\
    PyQt5\
    matplotlib\
    scipy\
    python-osc \
    PythonQwt

USER root
RUN apt-get update && \
    apt-get install -y\
    libx11-xcb1 libxcb1 libxcb-util1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-randr0 libxcb-render0 libxcb-render-util0 libxcb-shape0 libxcb-shm0 \
    libxcb-sync1 libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 libxkbcommon0 \
    libxkbcommon-x11-0 xvfb libgl1-mesa-glx 

#make workspace directory
RUN cd /home/appuser && \
    mkdir -p lumencontrol/src     

#install vscode server and extensions inside the container
#it propably won't work on computers of others because of the specific vscode version
#mostly stolen from here: https://gist.github.com/discrimy/6c21c10995f1914cf72cd8474d4501b2
#its great, because it means it is already installed into the image, so when starting a vscode instance inside the container, it will be already there.
#it will not have to download it.
#more info: https://github.com/microsoft/vscode-remote-release/issues/1718
USER root
RUN apt-get update && \
    apt-get install -y\
    curl \
    jq
#TODO put this at the top
RUN chown -R appuser:appuser /home/appuser

COPY  --chown=appuser:appuser ./misc/.devcontainer/ /home/appuser/lumencontrol/.devcontainer/
USER appuser
ARG VSCODE_COMMIT_HASH
RUN bash /home/appuser/lumencontrol/.devcontainer/preinstall_vscode.sh $VSCODE_COMMIT_HASH /home/appuser/lumencontrol/.devcontainer/devcontainer.json

RUN mkdir -p /home/appuser/lumencontrol/src/my_packages && \
    mkdir -p /home/appuser/lumencontrol/.vscode

#beat this!
USER appuser
RUN python3 -m pip install https://github.com/CPJKU/beat_this/archive/main.zip &&\
    python3 -m pip install -U tqdm einops soxr rotary-embedding-torch soundfile librosa

USER appuser
COPY --chown=appuser:appuser ./my_packages /home/appuser/lumencontrol/src/my_packages

#build the workspace
USER appuser
RUN cd /home/appuser/lumencontrol && \
    source /opt/ros/$ROS_DISTRO/setup.bash && \
    colcon build --symlink-install 

#make it that the setup.bash is sourced when a new terminal is opened
USER appuser
RUN echo "source /home/appuser/lumencontrol/install/setup.bash" >> /home/appuser/.bashrc


#natten

# USER appuser
# RUN cd tmp && git clone https://github.com/SHI-Labs/NATTEN && cd NATTEN && \
#     python3 -m pip install -r requirements.txt && \
#     python3 -m pip install ninja &&\
#     make CMAKE_PREFIX_PATH="/usr/local/lib/python3.10/dist-packages/torch/share/cmake" 
# # WITH_CUDA=1 CUDA_ARCH="8.0" CFLAGS="-01"

# #allin1 music analyzer
# USER appuser
# RUN python3 -m pip install git+https://github.com/CPJKU/madmom  &&\ 
#     python3 -m pip install allin1


USER appuser