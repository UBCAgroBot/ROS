#!/usr/bin/env bash
set -ex

echo 'installing cmake'

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null
apt-get update 
rm /usr/share/keyrings/kitware-archive-keyring.gpg 
apt-get install -y --no-install-recommends kitware-archive-keyring 
apt-cache policy cmake 
apt-get install -y --no-install-recommends cmake 
rm -rf /var/lib/apt/lists/* 
apt-get clean
cmake --version