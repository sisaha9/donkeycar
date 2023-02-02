from donkeycar import Vehicle
from donkeycar.parts.camera import OAKDCamera

V = Vehicle()

cam = OAKDCamera()
V.add(cam, threaded=True)

V.start()