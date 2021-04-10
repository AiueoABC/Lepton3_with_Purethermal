"""
AiueoABC Temperature Record System (Atres)
"""
import cv2
import numpy as np


class atres:
    def temp2color(self, int16KelvinArray, save=True):
        if int16KelvinArray is not np.ndarray:
            raise TypeError("atres.temp2png needs data as ndarray")
            # try:
            #     raise TypeError("atres.temp2png needs data as ndarray")
            # except TypeError as e:
            #     print(e)
        bStack, gStack = np.uint8(divmod(int16KelvinArray, 256))
        rStack = _raw_to_8bit(int16KelvinArray)
        atresimg = cv2.merge((bStack,gStack,rStack))
        if save:
            cv2.imwrite("colorTemperatureData.png", atresimg)
        return atresimg

    def bodytemp2color(self, int16KelvinArray, save=True):
        if int16KelvinArray is not np.ndarray:
            raise TypeError("atres.temp2png needs data as ndarray")
            # try:
            #     raise TypeError("atres.temp2png needs data as ndarray")
            # except TypeError as e:
            #     print(e)
        bStack, gStack = np.uint8(divmod(int16KelvinArray, 256))
        rStack = _raw_to_8bit_body(int16KelvinArray)
        atresimg = cv2.merge((bStack, gStack, rStack))
        if save:
            cv2.imwrite("AtresDataForBody.png", atresimg)
        return atresimg

    def atresimg2temp(self, atresimg):
        upperStack, lowerStack, _ = cv2.split(atresimg)
        int16KelvinArray = np.uint16(upperStack * 256 + lowerStack)
        return  int16KelvinArray


def _raw_to_8bit(data):
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(data, 8, data)
    return np.uint8(data)


def _raw_to_8bit_body(data):
    data = np.clip(data, 25000, 45000)
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(data, 8, data)
    return np.uint8(data)

