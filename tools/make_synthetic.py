"""Generate synthetic top-view 'cashier' videos (moving rectangle = operator).
No real data involved. usage: make_synthetic.py OUT.mp4 [duration] [absent_start absent_end]
scenarios via make(): absent=(t0,t1) operator absent; two=(t0,t1) second figure beside the operator"""
import sys
import cv2
import numpy as np


def make(path, dur=20, fps=30, absent=None, size=(640, 360), two=None):
    w, h = size
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(int(dur * fps)):
        t = i / fps
        fr = np.full((h, w, 3), 90, np.uint8)
        cv2.rectangle(fr, (50, 200), (590, 330), (60, 90, 140), -1)  # counter
        away = absent and absent[0] <= t < absent[1]
        if not away:  # standing figure (HOG-like silhouette is NOT guaranteed)
            cx = 320 + int(10 * np.sin(t))
            cv2.ellipse(fr, (cx, 120), (22, 28), 0, 0, 360, (200, 180, 160), -1)
            cv2.rectangle(fr, (cx - 35, 150), (cx + 35, 260), (40, 40, 160), -1)
        if two and two[0] <= t < two[1]:  # second figure, same colours, offset to the right
            cx2 = 470
            cv2.ellipse(fr, (cx2, 120), (22, 28), 0, 0, 360, (200, 180, 160), -1)
            cv2.rectangle(fr, (cx2 - 35, 150), (cx2 + 35, 260), (40, 40, 160), -1)
        # moving 'banknote' on counter (activity), always
        x = 100 + int(200 * (0.5 + 0.5 * np.sin(t * 2)))
        cv2.rectangle(fr, (x, 240), (x + 40, 260), (230, 230, 230), -1)
        vw.write(fr)
    vw.release()


if __name__ == "__main__":
    a = sys.argv
    make(a[1], float(a[2]) if len(a) > 2 else 20,
         absent=(float(a[3]), float(a[4])) if len(a) > 4 else None)
