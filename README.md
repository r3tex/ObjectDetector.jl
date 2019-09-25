# Object Detection in Julia using Flux
This package implements YOLOv3 and is written in pure Julia using the Flux machine learning framework. 

It loads both tiny and full YOLOv3 weights trained using Darknet code. It can run in realtime on a desktop GPU but on memory constrained devices like the Jetson TX2 it will spend a lot of time on memory allocation. Single image inference time is about 5 milliseconds on a GTX1080.

![screenshot](https://raw.githubusercontent.com/r3tex/ObjectDetector.jl/master/Screenie.PNG)
