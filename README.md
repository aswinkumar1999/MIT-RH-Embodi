# MIT-RH-Embodi
Repository for MIT Reality Hack - Team Embodi


Qualcomm -> HDMI Capture -> /dev/video4 -> PC -> Python ( CV ) -> AR 

Python ( OpenCV ) : 

Input ( /dev/video4 ) -> Pre-prcoessing  ( Resizing ) -> Yolo V5 Large -> Identifies if dog exists -> Rotate Servo if Dog edge of frame -> Pass Patch of dog in original frame to Emotion Clasifier ( Trained with `train_script.py` ) -> Get Confidence score of emotion -> Push to Flask Server


AR : 

Flask Server -> Send Emotion data as POST request -> AR Application -> Served by Python server 
