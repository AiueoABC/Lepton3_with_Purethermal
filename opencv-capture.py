#!/usr/bin/env python

"""
CV2 video capture example from Pure Thermal 1
"""

try:
    import cv2
except ImportError:
    print( "ERROR python-opencv must be installed")
    exit(1)
import numpy as np
import Atres

atres = Atres.atres()

class OpenCvCapture(object):
    """
    Encapsulate state for capture from Pure Thermal 1 with OpenCV
    """

    def __init__(self):
        # capture from the LAST camera in the system
        # presumably, if the system has a built-in webcam it will be the first
        for i in reversed(range(10)):
            print("Testing for presense of camera #{0}...".format(i))
            cv2_cap = cv2.VideoCapture(i)
            if cv2_cap.isOpened():
                cv2_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
                cv2_cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
                # cv2_cap.set(cv2.CAP_PROP_CONVERT_RGB, False)
                break

        if not cv2_cap.isOpened():
            print("Camera not found!")
            exit(1)

        self.cv2_cap = cv2_cap

    def show_video(self):
        """
        Run loop for cv2 capture from lepton
        """

        cv2.namedWindow("AtresImg", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Lepton Radiometry", cv2.WINDOW_NORMAL)
        print("Running, ESC or Ctrl-c to exit...")
        while True:
            ret, img = self.cv2_cap.read()
            if img.shape == (1, 39040):
                # 122*160*2
                a = np.reshape(img, [122, 320])
                img = np.array([a[:, 2 * i] + a[:, 2 * i + 1] for i in range(int(len(a[0]) / 2))]).T
            img = img[0:120, :]
            if ret == False:
                print("Error reading image")
                break
            if True:
                data = cv2.resize(img, (640, 480))
                atresimg = atres.bodytemp2color(data)
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
                img = raw_to_8bit(data)
                display_temperature(img, minVal, minLoc, (255, 0, 0))
                display_temperature(img, maxVal, maxLoc, (0, 0, 255))
                cv2.imshow('AtresImg', atresimg)
                cv2.imshow('Lepton Radiometry', img)
            else:
                cv2.imshow("Lepton Radiometry", cv2.resize(img, (640, 480)))
            if cv2.waitKey(5) == 27:
                break

        cv2.destroyAllWindows()


def raw_to_8bit(data):
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(data, 8, data)
    return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)


def display_temperature(img, val_k, loc, color):
    # val = ktof(val_k)
    val = ktoc(val_k)
    # cv2.putText(img, "{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(img, "{0:.1f} degC".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    x, y = loc
    cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
    cv2.line(img, (x, y - 2), (x, y + 2), color, 1)


def ktof(val):
    return (1.8 * ktoc(val) + 32.0)


def ktoc(val):
    return (val - 27315) / 100.0


if __name__ == '__main__':
    OpenCvCapture().show_video()
