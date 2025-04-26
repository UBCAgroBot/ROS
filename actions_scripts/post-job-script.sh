#!/bin/bash

rm -rf /home/runner/_temp/* || true
rm -rf /home/runner/_work/${{ github.repository }}/* || true

# should delete the old run markdown report files

# also should remove the model files from the save directory