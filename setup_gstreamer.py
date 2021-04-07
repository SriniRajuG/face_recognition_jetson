import cv2 as cv

width = 800
height = 600
flip = 2
gstreamer_pipe = f" \
    nvarguscamerasrc sensor-id=0 ! \
    video/x-raw(memory:NVMM), width=3264, height=2464, framerate=21/1, \
        format=NV12 ! \
    nvvidconv flip-method={flip} ! \
    video/x-raw, width={width}, height={height}, format=BGRx ! \
    videoconvert ! \
    video/x-raw, format=BGR ! \
    appsink \
    "

vid_cap = cv.VideoCapture(gstreamer_pipe)
while True:
    _, img = vid_cap.read()
    cv.imshow('PiCamera', img)
    cv.moveWindow('PiCamera', 0, 0)
    if cv.waitKey(1) == ord('q'):
        break
vid_cap.release()
cv.destroyAllWindows()
