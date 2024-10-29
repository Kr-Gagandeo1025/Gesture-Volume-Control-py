import cv2
import time
import handDetectionModule as htm
import math

#for audio control
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
 
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
   IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

cam = cv2.VideoCapture(0)
detector = htm.handDetector()
while True:
    ptime = time.time()
    #reading the videocapture
    success, img = cam.read()
    img = detector.findHands(img)
    lnList = detector.findPosition(img)


    if len(lnList) != 0:
        thumb = lnList[4]
        index = lnList[8]

        thumbY=thumb[2]
        indexY=index[2]
        thumbX=thumb[1]
        indexX=index[1]

        cv2.line(img,(thumbX,thumbY),(indexX,indexY),(0,0,255),2)
        lineLength=abs(math.sqrt((thumbX-indexX)**2 + (thumbY-indexY)**2))
        DefVol=-1*((156/lineLength)*10)
        # print(lineLength)
        try:
            if lineLength < 24:
                volume.SetMasterVolumeLevel(-56.0, None)
                cv2.putText(img, "Volume Set to Minimum at 2%", (10,100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
            elif lineLength > 156:
                volume.SetMasterVolumeLevel(-0.0, None)
                cv2.putText(img, "Volume Set to Maximum at 100%", (10,100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
            else :
                nVol = -1*((156/lineLength)*10)
                volume.SetMasterVolumeLevel(nVol, None)
        except:
            pass
    else:
        try:
           volume.SetMasterVolumeLevel(DefVol, None) 
        except:
            pass
        


    ctime= time.time()
    fps=1/(ctime-ptime)
    ptime=ctime

    cv2.putText(img, "fps="+str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)
