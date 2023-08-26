#!/bin/bash

#
# This scripts valid for Ubuntu Linux or any other Debian based Linux.
# Make sure to run `chmod +x install.sh` before running this script.
#

# Install Python 3
sudo apt install python3-pip

# Install Graphviz
sudo apt-get install graphviz

# Install required Python packages
pip install numpy torch torchvision matplotlib pandas torchviz
