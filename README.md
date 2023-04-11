# Google Inception v3 Video Object Detection

I worked on computer vision by using a pre-trained Google Inception V3 deep learning model to
search for objects in small video clips. I have deployed the model on Flask and allowed users to
upload videos within a given memory size. After a video is uploaded, the video is split into frames
and fed into the model to detect objects. A result page shows all detected objects in the frames and
their respective probabilities.

Deployed Web Application: http://objectdetection-env-2.eba-m988fzxm.us-west-2.elasticbeanstalk.com/