#!/bin/bash
#parameters from .param file
if ! [ -f "./misc/.params" ];
then
   cp ./misc/.params.example ./misc/.params
fi
source ./misc/.params
HOST_USER_GROUP_ARG=$(id -g $USER)
VSCODE_COMMIT_HASH=$(code --version | sed -n '2p')
echo $VSCODE_COMMIT_HASH
#export variable for building the image
HOST_USER_GROUP_ARG=$(id -g $USER)

docker build .\
    --tag $image_name:$image_tag \
    --build-arg HOST_USER_GROUP_ARG=$HOST_USER_GROUP_ARG \
    --build-arg VSCODE_COMMIT_HASH=$VSCODE_COMMIT_HASH \
    --build-arg SUB_VERSION=$SUB_VERSION \
    --build-arg ARCH=$ARCH \
    --build-arg BUILD_TYPE=$BUILD_TYPE \
   --progress=plain -t $image_name