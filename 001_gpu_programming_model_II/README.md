# is data transfer between CPU and GPU expensive?

Second the CUDA instructor John Owens, you want to minimize data transfer between CPU and GPU as much as you can, if you´re going to move a lot of data and do only a little bit of computation on that data, CUDA or GPU computing probably isn't a great fit for your problem.

# Student notes

The class notes here presented are not of my property except the exercices development and some additional notes added to professor slides, the class notes slides are here just for future fast reference purpouse and were extracted from Intro to Parallel Programming course created by Nvidia©, that is free available at Udacity©.
The main purpouse of this repo is to mantain an organized register of exercises source code and explain core functionality of algorithms involved using this readme.