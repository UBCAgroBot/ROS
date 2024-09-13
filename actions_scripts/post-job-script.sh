#!/bin/bash

rm -rf /home/runner/_temp/* || true
rm -rf /home/runner/_work/${{ github.repository }}/* || true