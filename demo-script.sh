#!/bin/bash
rm sample-filtered.wav
echo "press enter to play sample.wav"
read
mplayer sample.wav

echo "press enter to run classifier"
read
time python filter.py

echo "press enter to play sample-filtered.wav"
read
mplayer sample-filtered.wav
