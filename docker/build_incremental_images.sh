#!/bin/bash

bash build_base_images.sh

cd ..
docker build --network=host -t gadgetron_ubuntu_2004 .


