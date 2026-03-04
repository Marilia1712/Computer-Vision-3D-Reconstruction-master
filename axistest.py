import cv2 as cv
import numpy as np

# length of the axes in mm (same units as your world coordinates)
axis_length = 100.0
axis_3D = np.array([
    [0,0,0],                 # origin
    [axis_length,0,0],       # X axis
    [0,axis_length,0],       # Y axis
    [0,0,axis_length]        # Z axis
], dtype=np.float32)

# Loop over cameras
for cam in range(1, 5):
    # Load intrinsics
    fs_intr = cv.FileStorage(f"./cam{cam}_intrinsics.xml", cv.FILE_STORAGE_READ)
    K = fs_intr.getNode("CameraMatrix").mat()
    dist = fs_intr.getNode("DistortionCoeffs").mat()
    fs_intr.release()

    # Load extrinsics
    fs_ext = cv.FileStorage(f"./data/cam{cam}/config.xml", cv.FILE_STORAGE_READ)
    rvec = fs_ext.getNode("rvec").mat()
    tvec = fs_ext.getNode("tvec").mat()
    fs_ext.release()

    # Reshape
    K = K.reshape(3,3)
    dist = dist.reshape(-1,1)
    rvec = rvec.reshape(3,1)
    tvec = tvec.reshape(3,1)

    # Project axis into 2D
    imgpts, _ = cv.projectPoints(axis_3D, rvec, tvec, K, dist)
    imgpts = imgpts.reshape(-1,2).astype(int)

    # Read first frame of camera video
    cap = cv.VideoCapture(f"./data/cam{cam}/video.avi")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Camera {cam}: cannot read video frame.")
        continue

    # Draw axes: X=red, Y=green, Z=blue
    origin = tuple(imgpts[0])
    frame = cv.line(frame, origin, tuple(imgpts[1]), (0,0,255), 3)
    frame = cv.line(frame, origin, tuple(imgpts[2]), (0,255,0), 3)
    frame = cv.line(frame, origin, tuple(imgpts[3]), (255,0,0), 3)

    cv.imshow(f"Camera {cam} axes", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()