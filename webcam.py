#! /usr/bin/env python
# -*- coding:utf-8 -*-

import cv2


def main():

    ESC_KEY = 27
    INTERVAL = 33
    # FRAME_RATE = 30

    ORG_WINDOW_NAME = "org"
    # GAUSSIAN_WINDOW_NAME = "gaussian"

    DEVICE_ID = 0

    cascade_file = "./classifier/haarcascades/haarcascade_frontalface_alt2.xml"
    cascade = cv2.CascadeClassifier(cascade_file)

    cap = cv2.VideoCapture(DEVICE_ID)

    end_flag, c_frame = cap.read()
    height, width, hannels = c_frame.shape

    cv2.namedWindow(ORG_WINDOW_NAME)
    # cv2.namedWindow(GAUSSIAN_WINDOW_NAME)

    while end_flag == True:
        img = c_frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_list = cascade.detectMultiScale(img_gray, minSize=(100, 100))

        for (x, y, w, h) in face_list:

            color = (0, 0, 255)
            pen_w = 3
            # cv2.rectangle(img_gray, (x, y), (x+w, y+h), color, thickness=pen_w)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness=pen_w)

        cv2.imshow(ORG_WINDOW_NAME, c_frame)
        # cv2.imshow(GAUSSIAN_WINDOW_NAME, img_gray)

        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        end_flag, c_frame = cap.read()

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
