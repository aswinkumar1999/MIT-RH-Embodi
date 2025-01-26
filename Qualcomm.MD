setenforce 0
export XDG_RUNTIME_DIR=/dev/socket/weston
export WAYLAND_DISPLAY=wayland-1

# Camera Test
gst-launch-1.0 -e qtiqmmfsrc ! video/x-raw,format=NV12,width=1920,height=1080 ! waylandsink sync=false fullscreen=true

# YOLO V8 
gst-launch-1.0 -e --gst-debug=2 qtiqmmfsrc name=camsrc ! video/x-raw\(memory:GBM\),format=NV12,width=1920,height=1080,framerate=30/1 ! queue ! tee name=split split. ! queue ! qtimetamux name=metamux ! queue ! qtioverlay ! queue ! waylandsink sync=false fullscreen=true split. ! queue ! qtimlvconverter ! queue ! qtimltflite delegate=external external-delegate-path=libQnnTFLiteDelegate.so external-delegate-options=”QNNExternalDelegate,backend_type=htp;” model=/home/root/YOLOv8-Detection-Quantized.tflite ! queue ! qtimlvdetection threshold=75.0 results=10 module=yolov8m ! text/x-raw ! queue ! metamux

gst-launch-1.0 -e --gst-debug=2 qtiqmmfsrc name=camsrc ! video/x-raw\(memory:GBM\),format=NV12,width=1920,height=1080,framerate=30/1 ! queue ! tee name=split split. ! queue ! qtimetamux name=metamux ! queue ! qtioverlay ! queue ! waylandsink sync=false fullscreen=true split. ! queue ! qtimlvconverter ! queue ! qtimltflite delegate=external external-delegate-path=libQnnTFLiteDelegate.so external-delegate-options="QNNExternalDelegate,backend_type=htp;" model=/home/root/YOLOv8-Detection-Quantized.tflite ! queue ! qtimlvdetection threshold=70.0 results=10 module=yolov8 labels=/home/root/coco_labels.txt ! text/x-raw ! queue ! metamux

gst-launch-1.0 -e --gst-debug=3 \
qtiqmmfsrc name=camsrc ! video/x-raw\(memory:GBM\),format=NV12,width=1280,height=720,framerate=30/1 ! queue ! tee name=split \
split. ! queue ! qtimetamux name=metamux ! queue ! qtioverlay ! queue ! waylandsink sync=false fullscreen=true \
split. ! queue ! qtimlvconverter ! queue ! qtimltflite delegate=external external-delegate-path=libQnnTFLiteDelegate.so external-delegate-options="QNNExternalDelegate,backend_type=htp;" model=/home/root/YOLOv8-Detection-Quantized.tflite ! queue ! qtimlvdetection threshold=40.0 results=10 module=yolov8 labels=/home/root/coco_labels.txt constants="YOLOv8,q-offsets=<-107.0, -128.0, 0.0>,q-scales=<3.093529462814331, 0.00390625, 1.0>;" ! text/x-raw ! queue ! metamux.

gst-launch-1.0 -e --gst-debug=2 qtiqmmfsrc name=camsrc ! video/x-raw,format=NV12,width=1280,height=720,framerate=30/1,compression=ubwc ! queue ! tee name=split split. ! queue ! qtivcomposer name=mixer ! queue ! waylandsink fullscreen=true split. ! queue ! qtimlvconverter ! queue ! qtimltflite delegate=external external-delegate-path=libQnnTFLiteDelegate.so external-delegate-options="QNNExternalDelegate,backend_type=htp;" model=/home/root/YOLOv8-Detection-Quantized.tflite ! queue ! qtimlvdetection threshold=50.0 results=10 module=yolov8 labels=/opt/coco_labels.txt constants="YOLOv8,q-offsets=<-107.0, -128.0, 0.0>,q-scales=<3.093529462814331, 0.00390625, 1.0>;" ! video/x-raw,format=BGRA,width=640,height=360 ! queue ! mixer.
