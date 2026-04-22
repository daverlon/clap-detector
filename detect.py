import sys
import math

from ultralytics import YOLO
import torch

import pyqtgraph as pg
from PySide6 import QtWidgets

SAMPLES = 50
T = 3
TM = 30

def dx(data: list):
    l = len(data)
    ds = []
    for i in range(1, l):
        ds.append(data[i]-data[i-1])

    abs_ds = [abs(x) for x in ds]

    signed_dx = sum(ds)/(l-1)
    unsigned_dx = sum(abs_ds)/(l-1)

    return -signed_dx, unsigned_dx

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def clap_score(s_dx, u_dx):
    v = 0.010
    m = 0.010
    
    x = min(abs(s_dx) / v, 1.0)
    y = min(u_dx / m, 1.0)

    score = (x + y) / 2.0
    return score

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    win = pg.plot()
    win.setYRange(-0.05, 0.05, padding=0)
    win.setXRange(0, T*TM, padding=0)

    legend = win.addLegend() 

    s_curve = win.plot(pen="r", name="signed dx")
    u_curve = win.plot(pen="g", name="unsigned dx")

    model = YOLO("yolo26n-pose.pt")
    model.to("mps" if torch.backends.mps.is_available() else "cpu")

    results = model.predict(source=0, stream=True, show=True, verbose=False)

    data = []

    s_dx_data = []
    u_dx_data = []

    for result in results:
        try:
            xyn = result.keypoints.xyn  # normalized
            # print(xyn)

            left_shoulder_x, left_shoulder_y = xyn[0,5].cpu().numpy().tolist()
            right_shoulder_x, right_shoulder_y = xyn[0,6].cpu().numpy().tolist()
            shoulder_distance = euclidean_distance(left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y)

            left_wrist_x, left_wrist_y = xyn[0,11].cpu().numpy().tolist()
            # print(left_wrist_x, left_wrist_y)
            right_wrist_x, right_wrist_y = xyn[0,12].cpu().numpy().tolist()
            # print(right_wrist_x, right_wrist_y)

            delta = euclidean_distance(left_wrist_x, left_wrist_y, right_wrist_x, right_wrist_y)
            delta = delta / shoulder_distance

            # print(delta)
            if len(data) > SAMPLES: data = data[1:len(data)]
            data.append(delta)

            if (len(data)>=T):
                s_dx, u_dx = dx(data[len(data)-T:])
                s_dx_data.append(s_dx)
                u_dx_data.append(u_dx)

                score = clap_score(s_dx, u_dx)
                # print("Score:", score)
                if score > 0.7:
                    print("CLAPPING:", score)


            if len(s_dx_data) > T * TM: s_dx_data = s_dx_data[1:len(s_dx_data)]
            if len(u_dx_data) > T * TM: u_dx_data = u_dx_data[1:len(u_dx_data)]
            s_curve.setData(s_dx_data)
            u_curve.setData(u_dx_data)

            # curve.setData(data)
        except:
            continue
        

    sys.exit(app.exec())