import cv2
import time
import numpy as np
import tracking as tr
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

################################
largura_cam, altura_cam = 640, 480
################################

cam = cv2.VideoCapture(0)
cam.set(3, largura_cam)
cam.set(4, altura_cam)
previous_time = 0

detector = tr.DetectorDeMaos(detectionConfidence=0.7)

dispositivos = AudioUtilities.GetSpeakers()
interface = dispositivos.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
alcance_vol = volume.GetVolumeRange()
vol_min = alcance_vol[0]
vol_max = alcance_vol[1]
vol = 0
barra_vol = 400
vol_percentual = 0
while True:
    success, img = cam.read()
    img = detector.encontrarMaos(img)
    hand_landmarks = detector.encontrarPosicoes(img, draw=False)
    if len(hand_landmarks) != 0:
        # print(lista_lm[4], lista_lm[8])

        x1, y1 = hand_landmarks[4][1], hand_landmarks[4][2]
        x2, y2 = hand_landmarks[8][1], hand_landmarks[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(comprimento)

        # Faixa da mão 50 - 300
        # Faixa de volume -65 - 0

        vol = np.interp(length, [50, 300], [vol_min, vol_max])
        barra_vol = np.interp(length, [50, 300], [400, 150])
        vol_percentual = np.interp(length, [50, 300], [0, 100])
        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(barra_vol)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol_percentual)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    time_now = time.time()
    fps = 1 / (time_now - previous_time)
    previous_time = time_now
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    cv2.imshow("Imagem", img)
    cv2.waitKey(1)
