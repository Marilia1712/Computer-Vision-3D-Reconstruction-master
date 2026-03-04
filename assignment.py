from turtle import width
import glm
import cv2 as cv
import numpy as np
import os
import glob
import shutil
from skimage import measure  # marching_cubes

# ---- WORLD MAPPING CONFIG ----
# Your calibration extrinsics (rvec/tvec) are in the same units as the checkerboard square size
# used when calibrating (Assignment 1). Very often that is in *centimeters*, not millimeters.
# We'll infer scale using ||tvec|| magnitudes.
SCALE = 1.0
OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float32)
pattern_size = (8, 6)
block_size = 1.0 # voxel size
EDGE_SIZE = 115 # size of the square edge (in mm)

auto_objectPoints = {
    "cam1": [],
    "cam2": [],
    "cam3": [],
    "cam4": []
}
auto_imagePoints = {
    "cam1": [],
    "cam2": [],
    "cam3": [],
    "cam4": []
}
manual_objectPoints = {
    "cam1": [],
    "cam2": [],
    "cam3": [],
    "cam4": []
}
manual_imagePoints = {
    "cam1": [],
    "cam2": [],
    "cam3": [],
    "cam4": []
}
camera_matrix = {
    "cam1": None,
    "cam2": None,
    "cam3": None,
    "cam4": None
}
dist_coeffs = {
    "cam1": None,
    "cam2": None,
    "cam3": None,
    "cam4": None
}

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
top_left = (0,0)
top_right = (pattern_size[0] - 1, 0)
bottom_left = (0, pattern_size[1] - 1)
bottom_right = (pattern_size[0] - 1, pattern_size[1] - 1)


# built-in utility functions

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors

def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    return [[-64 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, -64 * block_size],
            [-64 * block_size, 64 * block_size, -64 * block_size]], \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]

def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations


# functions for corner selection and interpolation

def signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

def generate_board_points(cols, rows):
    """
    Function to define the board space.
    param: cols: number of columns in the grid
    param: rows: number of rows in the grid
    return: board_points: list containing the board poitns coordinates the given grid
    """
    board_tl = (0,0)
    board_tr = (cols - 1 , 0)
    board_br = (cols - 1, rows - 1)
    board_bl = (0, rows - 1)
    board_points = (board_tl, board_tr, board_br, board_bl)
    return np.array(board_points, dtype = np.float32)

def select_corners(event, x, y, flags, param):
    """
    UI interface to select the corners in the bad images.
    It generates a point where the mouse clicks and it shows with text which corner has to be selected.
    Uses the parameters needed by the cv.setMouseCallback funciton.
    """
    corners = ['top-left', 'top-right', 'bottom-right', 'bottom-left']
    font = cv.FONT_HERSHEY_SIMPLEX

    if event == cv.EVENT_LBUTTONDOWN:
        if len(param['2dpoints']) < 4:
            param['2dpoints'].append((x,y))
            cv.circle(param['base'], (x,y), 5, (255,0,0), -1)
    
    elif event == cv.EVENT_RBUTTONDOWN:
        param['2dpoints'].clear()
        param['base'] = param['original'].copy()

    param['display'] = param['base'].copy()

    n = len(param['2dpoints'])
    if n < 4:
        msg = f"Click {corners[n]} corner"    
    else:
        msg = f"Done! Press Enter to continue"
    
    cv.putText(param['display'], msg, (100,260), font, 1, (0, 0, 255), 2)
    cv.imshow('Image', param['display'])

def order_points(src, pattern_size):
    """
    Function to order points in the correct clockwise order,
    based on the angle between each manually added corner points

    """
    src = np.asarray(src, np.float32)
    cx = np.mean(src[:,0])
    cy = np.mean(src[:,1])
    angles = np.arctan2(src[:, 1] - cy, src[:, 0] - cx)
    order = np.argsort(angles)
    pts = src[order]

    if signed_area(pts) > 0:
        pts = pts[::-1]

    best_i = None
    best_y = 1e10
    for i in range(4):
        p = pts[i]
        q = pts[(i+1) % 4]
        y_avg = 0.5 * (p[1] + q[1])
        if y_avg < best_y:
            best_y = y_avg
            best_i = i
    p_top0 = pts[best_i]
    p_top1 = pts[(best_i + 1) % 4]

    if p_top0[0] < p_top1[0]:
        top_left, top_right = p_top0, p_top1
        bottom_right = pts[(best_i + 2) % 4]
        bottom_left = pts[(best_i + 3) % 4]
    else:
        top_left, top_right = p_top1, p_top0
        bottom_right = pts[(best_i + 3) % 4]
        bottom_left = pts[(best_i + 2) % 4]
    
    ordered_points = np.array([top_left, top_right, bottom_right, bottom_left], np.float32)

    expected = (pattern_size[0] - 1) / (pattern_size[1] - 1)

    len_x = np.linalg.norm(ordered_points[1] - ordered_points[0])  # top right, top left
    len_y = np.linalg.norm(ordered_points[3] - ordered_points[0])  # bottom left, top left
    observed = len_x / (len_y + 1e-9)
    if observed < 1.0 and expected > 1.0:
        ordered_points = np.array([ordered_points[0], ordered_points[3], ordered_points[2], ordered_points[1]], np.float32)
    return ordered_points

def interpolate_corners(points, cols, rows):
    """
    Function that performs interpolation of image points.
    param: points: list of four points of coordinates that we want to interpolate
    param: cols: number of columns in the grid
    param: rows: number of rows in the grid
    return: img_grid: new grid containing all cols*rows set of points
    """
    if len(points) != 4:
        raise Exception("Four points should be selected.")
    img_points = np.array(points, dtype = np.float32)
    board_points = generate_board_points(cols, rows)
    H = cv.getPerspectiveTransform(board_points, img_points)
    all_points = []
    for y in range(0, rows):
        for x in range(0, cols):
            all_points.append((x,y))
    all_points = np.array(all_points, dtype = np.float32)
    img_grid = cv.perspectiveTransform(all_points.reshape(cols*rows, 1, 2), H)
    return img_grid




"""
#################################################################################################################################
"""
"""
    Task 1 + Choice Task 1 + Choice Task 7: (improved) calibration and automated rejection of low-quality calibration frames 
"""

MIN_POINTS = (pattern_size[0]*pattern_size[1]) // 4 ## min number of points needed for the convex hull technique

# functions for camera calibration
def calibrate_camera(cameraNumber, video_path, dest, dest_manual, frame_number = 50):

    cap = cv.VideoCapture(video_path)

    manual_images = glob.glob(f'{dest_manual}/*.jpg')
    video_length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    step = video_length // frame_number
    i = 0
    crns = {}

    shutil.rmtree(dest, ignore_errors=True)
    shutil.rmtree(dest_manual, ignore_errors=True)
    os.makedirs(dest_manual)
    os.makedirs(dest)

    while True:
        ret_, frame = cap.read()
        if not ret_:
            break
        i += 1


        if i % step == 0:

            if i > video_length:
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (pattern_size[0],pattern_size[1]), None)

            # case 1: straighforward localization of corners
            if ret == True:
                print(f"\nFrame {i}: chessboard detected")

                corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                h, w = gray.shape

                # ------- Choice Task 7: - Coverage check           -> TODO: move this to underneath the three cases
                xs = corners2[:,0,0]
                ys = corners2[:,0,1]
                cov_x = (xs.max() - xs.min()) / w
                cov_y = (ys.max() - ys.min()) / h

                if cov_x < 0.2 or cov_y < 0.2:
                    print("Rejected: spatial coverage is not enough")
                    continue

                # ------ Choice Task 7: - Board area check
                area = cv.contourArea(corners2.reshape(-1,2))
                img_area = w * h
                area_ratio = area / img_area

                print(f"  board area ratio = {area_ratio:.3f}")

                if area_ratio < 0.0025: #empirical threshold
                    print("Rejected: board too small/far")
                    continue
                
                # if frame is fine
                print("Accepted")

                auto_objectPoints[f"cam{cameraNumber}"].append(board_object_points)
                auto_imagePoints[f"cam{cameraNumber}"].append(corners2)

                cv.drawChessboardCorners(frame, (pattern_size[0],pattern_size[1]), corners2, ret)
                cv.imshow('img', frame)
                cv.waitKey(200)

                crns[f"frame_{i}_{video_path[10:-5]}"] = corners2

                # Draw and display the corners
                cv.drawChessboardCorners(frame, (pattern_size[0],pattern_size[1]), corners2, ret)
                cv.imshow('img', frame)
                cv.waitKey(500)
                crns[f"frame_{i}_{video_path[10:-5]}"] = corners2

            # case 2: partial localization of corners -> Choice task 1 - Convex Hull technique
            elif corners is not None and len(corners) >= MIN_POINTS:

                pts = corners.reshape(-1,2).astype(np.float32)
                hull = cv.convexHull(pts)
                rect = cv.minAreaRect(hull)
                box  = cv.boxPoints(rect)
                src = np.array(box, dtype=np.float32)

                # enforce order
                ordered = order_points(src, pattern_size = pattern_size)
                ordered_ref = ordered.reshape(-1, 1, 2).astype(np.float32)
                ordered_refined = ordered_ref.reshape(4, 2)

                dst = np.float32([
                    top_left,
                    top_right,
                    bottom_right,
                    bottom_left
                ])
                W = (pattern_size[0] - 1) * 100
                H = (pattern_size[1] - 1) * 100
                
                #warping for better corner estimation
                H_warp = cv.getPerspectiveTransform(ordered_refined, dst)
                warped = cv.warpPerspective(img, H_warp, (W, H))
                gray_warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)

                # interpolation
                warped_grid = interpolate_corners(dst, pattern_size[0], pattern_size[1])
                warped_grid = warped_grid.reshape(-1, 1, 2).astype(np.float32)

                # inverse warping to go back to original image
                H_inv = cv.getPerspectiveTransform(dst, ordered_refined)
                img_grid = cv.perspectiveTransform(warped_grid, H_inv)

                img_points = img_grid.reshape(-1, 2)
                print("Using convex hull fallback")

                # Visualization for convex hull fallback
                frame_hull = frame.copy()

                # draw convex hull in green
                cv.polylines(frame_hull, [hull.astype(np.int32)], isClosed=True, color=(0,255,0), thickness=2)

                # draw minAreaRect rectangle in red
                rect_pts = box.astype(np.int32)
                cv.polylines(frame_hull, [rect_pts], isClosed=True, color=(0,0,255), thickness=2)

                # draw the 4 ordered points in blue
                for (x, y) in ordered_refined:
                    cv.circle(frame_hull, (int(x), int(y)), 5, (255,0,0), -1)

                cv.putText(frame_hull, f"Convex Hull Fallback - Frame {i}", (10,30),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

                cv.imshow("Convex Hull Visualization", frame_hull)
                cv.waitKey(5000)  # show for 500 ms, adjust as needed

            # case 3: insufficient/absent localization of corners -> manual input
            else:
                img = frame.copy()
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                fname = f'frame_{i}_{video_path[11:-5]}.jpg'

                state = {
                    "original": img.copy(),
                    "base": img.copy(),
                    "display": img.copy(),
                    "interpolation": img.copy(),
                    "2dpoints": [],
                    "3dpoints": [],
                    "filename": fname
                }

                cv.imshow('Image', state['display'])
                cv.setMouseCallback("Image", select_corners, state)
                
                key = cv.waitKey(0)

                if key == 27:
                    print("Exit")
                    cv.destroyAllWindows()
                    print(state['2dpoints'])
                    break

                if key == ord('s'):
                    print("Skipped image: ", fname)
                    continue

                if len(state['2dpoints']) != 4:
                    print(f'This image was skipped : {fname}. Please select at least 4 points in the next one.')
                    continue

                src = np.array(state['2dpoints'], dtype=np.float32)

                # enforce order 
                ordered = order_points(src, pattern_size = pattern_size)
                ordered_ref = ordered.reshape(-1, 1, 2).astype(np.float32)

                
                cv.cornerSubPix(gray, ordered_ref, (11,11), (-1,-1), criteria)
                ordered_refined = ordered_ref.reshape(4, 2)

                dst = np.float32([
                    top_left,
                    top_right,
                    bottom_right,
                    bottom_left
                ])
                W = (pattern_size[0] - 1) * 100
                H = (pattern_size[1] - 1) * 100
                
                # warping for better corner estimation
                H_warp = cv.getPerspectiveTransform(ordered_refined, dst)
                warped = cv.warpPerspective(img, H_warp, (W, H))
                gray_warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)

                # interpolation
                warped_grid = interpolate_corners(dst, pattern_size[0], pattern_size[1])
                warped_grid = warped_grid.reshape(-1, 1, 2).astype(np.float32)

                # inverse warping to go back to original image
                H_inv = cv.getPerspectiveTransform(dst, ordered_refined)
                img_grid = cv.perspectiveTransform(warped_grid, H_inv)

                img_points = img_grid.reshape(-1, 2)
                manual_imagePoints[f"cam{cameraNumber}"].append(img_points)
                manual_objectPoints[f"cam{cameraNumber}"].append(board_object_points.copy())

                for pt in img_grid:
                    x, y = pt[0]
                    cv.circle(state['interpolation'], (int(x), int(y)), 5, (255, 0, 0), -1)

                cv.imshow('Image', state["interpolation"])
                new_path = os.path.join("new_images", os.path.basename(fname))
                cv.imwrite(new_path, state["interpolation"])

                cv.waitKey(0)
                cv.destroyAllWindows()
                crns[f'frame_{i}_{video_path[10:-5]}'] = img_points.reshape(-1,1,2)

    print(crns)
    return crns

def create_object_points(cols, rows, square_size_mm = EDGE_SIZE):
    """
    Create the 3D world coordinates of the chessboard corners.
    """
    objp = []

    for y in range(rows):
        for x in range(cols):
            X = x * square_size_mm
            Y = y * square_size_mm
            Z = 0.0
            objp.append((X, Y, Z))

    return np.array(objp, dtype=np.float32)

def save_camera_config_xml(filename, K, dist, rvec=None, tvec=None):
    """
    Save the camera configuration at the end of calibration in an xml file
    """
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)

    fs.write("CameraMatrix", K.astype(np.float32))
    fs.write("DistortionCoeffs", dist.astype(np.float32))

    if rvec is not None:
        fs.write("rvec", rvec.astype(np.float32))
    if tvec is not None:
        fs.write("tvec", tvec.astype(np.float32))

    fs.release()

# functions for improved camera calibration (Choice task 7)
def reproj_error(objp, imgp, rvec, tvec, K, dist):
    proj, _ = cv.projectPoints(objp, rvec, tvec, K, dist)
    return np.mean(np.linalg.norm(proj.reshape(-1,2) - imgp.reshape(-1,2), axis=1))

def calibrate_with_reprojection_filter(objectPoints,
                                       imagePoints,
                                       image_size,
                                       criteria,
                                       threshold=1.0,
                                       verbose=True):
    """
    Runs calibration, computes per-frame reprojection error,
    rejects bad frames, and recalibrates.

    Returns:
        cameraMatrix, distCoeffs, rvecs, tvecs
    """

    # standard calibration
    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
        objectPoints, imagePoints, image_size, None, None,
        flags=0, criteria=criteria
    )
    print("\n=== Per-frame reprojection errors ===")

    good_obj = []
    good_img = []
    errors = []

    # reprojection error per frame
    for idx, (objp, imgp, rvec, tvec) in enumerate(zip(objectPoints, imagePoints, rvecs, tvecs)):

        proj, _ = cv.projectPoints(objp, rvec, tvec, K, dist)

        err = np.mean(
            np.linalg.norm(
                proj.reshape(-1, 2) - imgp.reshape(-1, 2),
                axis=1
            )
        )
        errors.append(err)

        if verbose:
            print(f"Frame {idx:02d} → error = {err:.3f}px") #print errors per frmae

        if err < threshold:
            good_obj.append(objp)
            good_img.append(imgp)
        else:
            if verbose:
                print("Rejected")

    # if no frame gets rejected, no need for calibrating again
    if len(good_obj) == len(objectPoints):
        if verbose:
            print("\nNo frames rejected.")
        return K, dist, rvecs, tvecs
    
    # otherwise recalibrate using the good frames
    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
        good_obj, good_img, image_size, None, None,
        flags=0, criteria=criteria
    )
    return K, dist, rvecs, tvecs


board_object_points = create_object_points(pattern_size[0], pattern_size[1], EDGE_SIZE)

#1.1 loop through the cameras to calibrate them (intrinsic parameters estimation)
for cameraNumber in range(1, 5):

    dest_man = f'./manual_images/cam{cameraNumber}'
    dest = f'./images/cam{cameraNumber}'
    video_path = f'data/cam{cameraNumber}/intrinsics.avi'
    crns = calibrate_camera(cameraNumber, video_path, dest, dest_man)

    # read image size 
    images = glob.glob(f'./images/cam{cameraNumber}/*.jpg')
    image_size = (cv.imread(images[0]).shape[1], cv.imread(images[0]).shape[0]) 
    #Run images
    objectPoints = auto_objectPoints[f"cam{cameraNumber}"] + manual_objectPoints[f"cam{cameraNumber}"]
    imagePoints  = auto_imagePoints[f"cam{cameraNumber}"]  + manual_imagePoints[f"cam{cameraNumber}"]

    # reprojection and recalibration (choice task 7) 
    cameraMatrix, distCoeffs, rvecs, tvecs = calibrate_with_reprojection_filter(
        objectPoints,
        imagePoints,
        image_size,
        criteria,
        threshold=1.0,   #reject calibration with avg corner reprojection >= 1px
        verbose=True
    )

    camera_matrix[f"cam{cameraNumber}"] = cameraMatrix
    dist_coeffs[f"cam{cameraNumber}"] = distCoeffs

    #1.3 save the intrinsic parameters in a text file for each camera
    with open(f'cam{cameraNumber}.txt', 'w') as f:
        f.write(f"Camera Matrix:\n{camera_matrix[f'cam{cameraNumber}']}\n\nDistortion Coefficients:\n{dist_coeffs[f'cam{cameraNumber}']}")


#1.2 estrinsic parameters 
for cameraNumber in range(1, 5):
    extrinsic_path = f'data/cam{cameraNumber}/checkerboard.avi'
    dest_man = f'./manual_images/cam{cameraNumber}_ext'
    dest = f'./images/cam{cameraNumber}_ext'
    crns = calibrate_camera(cameraNumber, extrinsic_path, dest, dest_man)
    K = camera_matrix[f"cam{cameraNumber}"]
    dist = dist_coeffs[f"cam{cameraNumber}"]
    save_camera_config_xml(f"cam{cameraNumber}_intrinsics.xml", K, dist)
    objp = create_object_points(pattern_size[0],pattern_size[1],EDGE_SIZE)

    best = None
    for key, corners in crns.items():
        ok, rvec, tvec = cv.solvePnP(objp, corners, K, dist)
        if not ok:
            continue
        proj, _ = cv.projectPoints(objp, rvec, tvec, K, dist)
        err = np.mean(np.linalg.norm(proj.reshape(-1,2) - corners.reshape(-1,2), axis=1))
        if best is None or err < best[0]:
            best = (err, key, rvec, tvec)
    if best is None:
        print(f"cam{cameraNumber}: no valid PnP solution (no detected / manual corners).")
        continue


    err, key, rvec, tvec = best
    print(f"cam{cameraNumber}: Using {key}, reproj err={err:.3f}px")
    print("rvec:\n", rvec)
    print("tvec:\n", tvec)
    save_camera_config_xml(f"cam{cameraNumber}_config.xml", K, dist, rvec, tvec)

"""
#################################################################################################################################
"""
"""
    Task 2: Background Subtraction
"""


def background_detection(video_path, k, n_train):
    """ 
    Function that implements background detection using gaussian mixture mdoel.
    param: video_path (str): video path to the background video
    param: k (int): number of gaussian components
    param: n_train (int): number of frames used in training
    return: 
    """
    back = cv.VideoCapture(video_path)
    n_frames = back.get(cv.CAP_PROP_FRAME_COUNT)
    print("Number of frames: ", n_frames)

    #take half of the frames
    #frames_indeces = np.linspace(0, n_frames, int(n_frames//2), dtype = np.int32)
    #print("Frames considered: ", len(frames_indeces))

    # background subtractor gaussian
    sub = cv.createBackgroundSubtractorMOG2(
        history=n_train,
        varThreshold = 16,
        detectShadows = False
    )
    sub.setNMixtures(k)
    lr_train = 1.0/float(n_train)

    trained = 0
    while trained < n_train:
        ret, frame = back.read()
        if not ret:
            break
        #frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        _ = sub.apply(frame, learningRate = lr_train)
        trained += 1
    
    back.release()
    return sub

def get_thresholds(video_path, img_path, sub, frame_idx):
    back_bgr = sub.getBackgroundImage()
    back_hsv = cv.cvtColor(back_bgr, cv.COLOR_BGR2HSV)
    cap = cv.VideoCapture(video_path)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    manual = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    manual = cv.resize(manual, (frame_hsv.shape[1], frame_hsv.shape[0]))
    manual = ((manual > 127).astype(np.uint8)) * 255

    hue_diff = cv.absdiff(frame_hsv[:, :, 0], back_hsv[:, :, 0])
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)

    sat_diff = cv.absdiff(frame_hsv[:, :, 1], back_hsv[:, :, 1])
    val_diff = cv.absdiff(frame_hsv[:, :, 2], back_hsv[:, :, 2])  # <-- was wrong in your code

    #best_iou = -1
    best_score = -float("inf")
    best = (0, 0, 0)
    #best = (None, None, None)
    ellipse3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))


    for thresh_hue in range(0,61,2):
        for thresh_sat in range(0,101,5):
            for thresh_val in range(0,101,5):
                mask_hue = (hue_diff > thresh_hue).astype(np.uint8) * 255
                mask_sat = (sat_diff > thresh_sat).astype(np.uint8) * 255
                mask_val = (val_diff > thresh_val).astype(np.uint8) * 255

                pred = cv.bitwise_or(mask_hue, mask_sat)
                pred = cv.bitwise_or(pred, mask_val)
                pred = cv.morphologyEx(pred, cv.MORPH_OPEN, ellipse3, iterations=1)

                # inter = np.logical_and(pred > 0, manual > 0).sum()
                # uni = np.logical_or(pred > 0, manual > 0).sum()
                # iou = (inter / uni) if uni > 0 else 0.0

                # if iou > best_iou:
                #     best_iou = iou
                #     best = (thresh_hue, thresh_sat, thresh_val)
                

                error = cv.bitwise_xor(pred, manual)
                score = -np.count_nonzero(error)   # higher is better (less error)

                if score > best_score:
                    best_score = score
                    best = (thresh_hue, thresh_sat, thresh_val)
    return best, best_score
    #best_iou

def postprocess(mask, kernel_size=5, open_it=1, close_it=2, min_area=500):
    mask = (mask > 0).astype(np.uint8) * 255
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if open_it > 0:
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k, iterations=open_it)
    if close_it > 0:
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k, iterations=close_it)

    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)

    valid_ids = []
    for i in range(1, num_labels):
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            valid_ids.append(i)

    if not valid_ids:
        return np.zeros_like(mask)

    areas = np.array([stats[i, cv.CC_STAT_AREA] for i in valid_ids])
    biggest = valid_ids[int(np.argmax(areas))]

    clean = np.zeros_like(mask)
    clean[labels == biggest] = 255
    return clean

def background_subtraction(video_path, sub, thresh_hue, thresh_sat, thresh_val, kernel):
    back_bgr = sub.getBackgroundImage()
    back_hsv = cv.cvtColor(back_bgr, cv.COLOR_BGR2HSV)

    cap = cv.VideoCapture(video_path)
    masks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        hue_diff = cv.absdiff(frame_hsv[:, :, 0], back_hsv[:, :, 0])
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)

        sat_diff = cv.absdiff(frame_hsv[:, :, 1], back_hsv[:, :, 1])
        val_diff = cv.absdiff(frame_hsv[:, :, 2], back_hsv[:, :, 2]) 

        mask_hue = (hue_diff > thresh_hue).astype(np.uint8) * 255
        mask_sat = (sat_diff > thresh_sat).astype(np.uint8) * 255
        mask_val = (val_diff > thresh_val).astype(np.uint8) * 255

        mask = cv.bitwise_or(mask_hue, mask_sat)
        mask = cv.bitwise_or(mask, mask_val)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
        mask = postprocess(mask, kernel_size=5, open_it=1, close_it=2, min_area=800)
        masks.append(mask)

    cap.release()
    return masks

def mask_from_thresholds_at_frame(video_path, sub, frame_idx, thresh_hue, thresh_sat, thresh_val, kernel):
    back_bgr = sub.getBackgroundImage()
    back_hsv = cv.cvtColor(back_bgr, cv.COLOR_BGR2HSV)

    cap = cv.VideoCapture(video_path)
    cap.set(cv.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    hue_diff = cv.absdiff(frame_hsv[:, :, 0], back_hsv[:, :, 0])
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)
    sat_diff = cv.absdiff(frame_hsv[:, :, 1], back_hsv[:, :, 1])
    val_diff = cv.absdiff(frame_hsv[:, :, 2], back_hsv[:, :, 2])

    mask_hue = (hue_diff > thresh_hue).astype(np.uint8) * 255
    mask_sat = (sat_diff > thresh_sat).astype(np.uint8) * 255
    mask_val = (val_diff > thresh_val).astype(np.uint8) * 255

    mask = cv.bitwise_or(mask_hue, mask_sat)
    mask = cv.bitwise_or(mask, mask_val)

    #mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    mask = postprocess(mask, kernel_size=5, open_it=1, close_it=2, min_area=800)
    #mask = cv.erode(mask, cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)), iterations=1)

    return mask


"""
#################################################################################################################################
"""
"""
    Task 3: Voxel Reconstruction

"""

# Get camera position (adaptation of code from Assingnment 1)

cam_pos, cam_orient, cam_R = {}, {}, {}
rvecs, tvecs, cam_matrix, dist_coeff, params = {}, {}, {}, {}, {}

for i in range(1, 5):

    fs = cv.FileStorage(f"data/cam{i}/config.xml", cv.FILE_STORAGE_READ)
    rvec = fs.getNode("rvec").mat()
    tvec = fs.getNode("tvec").mat()
    camera_matrix = fs.getNode("CameraMatrix").mat()
    dist = fs.getNode("DistortionCoeffs").mat()
    fs.release()

    rvec = rvec.reshape(3,1)
    tvec = tvec.reshape(3,1)
    dist = dist.reshape(5,1)
    camera_matrix = camera_matrix.reshape(3,3)
    rvecs[f'cam{i}'] = rvec
    tvecs[f'cam{i}'] = tvec
    cam_matrix[f"cam{i}"] = camera_matrix
    dist_coeff[f"cam{i}"] = dist 

    R, _ = cv.Rodrigues(rvec)

    cam_position = (-R.T @ tvec).reshape(3,)

    cam_direction = (R.T @ np.array([0, 0, 1.0])).reshape(3,)
    cam_direction = cam_direction / np.linalg.norm(cam_direction)

    cam_pos[f"cam{i}"] = cam_position
    cam_orient[f"cam{i}"] = cam_direction
    cam_R[f"cam{i}"] = R

    #Store params for each camera
    params[f'cam{i}'] = [camera_matrix, dist, rvec, tvec]

def find_voxel_centres(width, height, depth, bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    dx = (xmax - xmin)/ width
    dy = (ymax - ymin) / height
    dz = (zmax - zmin) / depth

    centres = np.zeros((height, width, depth, 3), dtype=np.float32)

    for j in range(height):
        for i in range(width):
            for k in range(depth):
                x = xmin + (i + 0.5) * dx
                y = ymin + (j + 0.5) * dy
                z = zmin + (k + 0.5) * dz
                centres[j, i, k] = (x, y, z)

    return centres, (dx, dy, dz)

def construct_lookup(centres, params, img_size):
    lut = {
        'cam1':{'uv': [], 'valid': []},
        'cam2':{'uv': [], 'valid': []},
        'cam3':{'uv': [], 'valid': []},
        'cam4':{'uv': [], 'valid': []},       
    }
    n = centres.shape[0] * centres.shape[1] * centres.shape[2]
    centres_flat = centres.reshape(n, 1, 3).astype(np.float32)

    for cam in range(1,5):
        camera_matrix, dist,rvec, tvec = params[f"cam{cam}"]
        image_points, _ = cv.projectPoints(centres_flat, rvec, tvec, camera_matrix, dist)
        uv = np.rint(image_points).reshape(n, 2).astype(np.int32)
        uv = uv.astype(np.int32)
        w,h = img_size[cam]
        valid = np.ones(n, dtype=bool)
        R, _ = cv.Rodrigues(rvec)
        Xc = (R @ centres_flat.reshape(n, 3).T + tvec).T #voxels in camrea coordinates
        Zc = Xc[:, 2] #depth of voxel
        
        for p in range(n):
            u = uv[p,0]
            v = uv[p,1]

            if u < 0 or u >= w or v<0 or v >= h:
                valid[p] = False
                uv[p] = (-1,-1)
            elif Zc[p] <= 0:
                valid[p] = False
                uv[p] = (-1,-1)
            else:
                valid[p] = True
                
        lut[f'cam{cam}']['uv'] = uv
        lut[f'cam{cam}']['valid'] = valid
    for cam in range(1, 5):
        valid_ratio = lut[f"cam{cam}"]["valid"].mean()
        print(f"cam{cam} valid ratio: {valid_ratio:.3f}")
    return lut

def get_mask_for_frame(frame_index, bg_sub, thresholds, kernel):
    mask_cam = {}

    for cam in range(1, 5):
        th_h, th_s, th_v = thresholds[cam]
        mask_cam[cam] = mask_from_thresholds_at_frame(
            video_path=f"data/cam{cam}/video.avi",
            sub=bg_sub[cam],
            frame_idx=frame_index,
            thresh_hue=th_h,
            thresh_sat=th_s,
            thresh_val=th_v,
            kernel=kernel
        )

    # print fg areas AFTER masks are computed
    for cam in range(1, 5):
        m = mask_cam[cam]
        if m is None:
            print(f"cam{cam} fg_area=None (mask None)")
        else:
            print(f"cam{cam} fg_area={int(np.count_nonzero(m))}")

    return mask_cam

def remove_voxels(lut, masks, rule="AND"):
    # n = number of voxels (use cam1 as reference)
    n = len(lut['cam1']['valid'])

    active = np.ones(n, dtype=bool)

    for p in range(n):
        votes = 0
        seen = 0
        killed = False

        for cam in range(1, 5):
            if not lut[f'cam{cam}']['valid'][p]:
                continue
            
            u, v = lut[f'cam{cam}']['uv'][p]
            if u < 0 or v < 0:
                continue
            seen += 1

            if masks[cam][v, u] > 0:
                votes += 1
            else:
                if rule == "AND":
                    active[p] = False
                    killed = True
                    break

        if killed:
            continue
        if seen < 2:
            active[p] = False
        elif rule == "AND":
            active[p] = (votes == seen)
        elif rule == "MAJORITY":
            active[p] = (votes >= (seen // 2 + 1))

        # if rule == "AND":
        #     active[p] = (seen > 0 and votes == seen)
        # elif rule == "MAJORITY":
        #     active[p] = (votes >= 2.5)
    
    print("Active voxels:", active.sum(), "out of", len(active))

    return active

def to_gl_coord(X_calib):
    X_gl = np.array([X_calib[0], X_calib[2], X_calib[1]]) * SCALE + OFFSET
    return X_gl

def render_lists(centres, active):
    pos = []
    colors = []

    centres_flat = centres.reshape(-1, 3)
    n = len(centres_flat)

    for p in range(n):
        if active[p]:
            Xgl = to_gl_coord(centres_flat[p])
            pos.append([Xgl[0], Xgl[1], Xgl[2]])
            colors.append([1.0, 1.0, 1.0])

    return pos, colors
        

def visualize_masks_all_cams(frame_idx=0, k=5, n_train=200):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    bg_sub = {}
    thresholds = {}

    for cam in [1, 2, 3, 4]:
        bg_sub[cam] = background_detection(f"data/cam{cam}/background.avi", k=k, n_train=n_train)

        candidates = [
            f"data/cam{cam}/manual.png",
            f"data/cam{cam}/mask.png",
            f"data/cam{cam}/segmentation.png",
            f"data/cam{cam}/foreground.png",
        ]
        img_path = next((p for p in candidates if os.path.exists(p)), None)

        if img_path is None:
            thresholds[cam] = (10, 25, 25)
        else:
            thresholds[cam], best_iou = get_thresholds(
                f"data/cam{cam}/video.avi",
                img_path,
                bg_sub[cam],
                frame_idx=frame_idx
            )
            print(f"cam{cam} best thresholds={thresholds[cam]} IoU={best_iou:.4f}")

    for cam in [1, 2, 3, 4]:
        video_path = f"data/cam{cam}/video.avi"
        cap = cv.VideoCapture(video_path)
        cap.set(cv.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"cam{cam}: could not read frame {frame_idx}")
            continue

        th_h, th_s, th_v = thresholds[cam]
        mask = mask_from_thresholds_at_frame(
            video_path,
            bg_sub[cam],
            frame_idx=frame_idx,
            thresh_hue=th_h,
            thresh_sat=th_s,
            thresh_val=th_v,
            kernel=kernel
        )
        if mask is None:
            print(f"cam{cam}: mask is None")
            continue

        overlay = frame.copy()
        overlay[mask > 0] = (0, 255, 0)

        cv.imshow(f"cam{cam} frame", frame)
        cv.imshow(f"cam{cam} mask", mask)
        cv.imshow(f"cam{cam} overlay", overlay)

    key = cv.waitKey(0) & 0xFF
    cv.destroyAllWindows()
    return key

# -------------------------
# Global cache for OpenGL calls
# -------------------------
_CACHE = {
    "initialized": False,
    "frame_idx": 0,
    "centres": None,
    "lut": None,
    "img_size": None,
    "bg_sub": None,
    "thresholds": None,
    "kernel": None,
    "bounds": None,
}

def _init_voxel_system(width, height, depth):
    # 1) choose bounds in calibration units (TUNE THESE)
    bounds = (
        0,    3800,     # x
        -2000, 2100,    # y  (più giù per i piedi)
        -2300, 2100     # z
    )  

    # 2) image sizes
    img_size = {}
    for cam in range(1, 5):
        cap = cv.VideoCapture(f"data/cam{cam}/video.avi")
        ret, frame0 = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read first frame for cam{cam}")
        H, W = frame0.shape[:2]
        img_size[cam] = (W, H)

    # 3) centres + LUT
    centres, _ = find_voxel_centres(width, height, depth, bounds)
    lut = construct_lookup(centres, params, img_size)

    # 4) background models + thresholds
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    bg_sub = {}
    thresholds = {}
    for cam in range(1, 5):
        bg_sub[cam] = background_detection(f"data/cam{cam}/background.avi", k=5, n_train=200)

        # optional: compute thresholds from manual if present
        candidates = [
            f"data/cam{cam}/manual.png",
            f"data/cam{cam}/mask.png",
            f"data/cam{cam}/segmentation.png",
            f"data/cam{cam}/foreground.png",
        ]
        img_path = next((p for p in candidates if os.path.exists(p)), None)
        if img_path is None:
            thresholds[cam] = (10, 25, 25)
        else:
            thresholds[cam], _ = get_thresholds(f"data/cam{cam}/video.avi", img_path, bg_sub[cam], frame_idx=0)

    # 5) set SCALE/OFFSET mapping to OpenGL (rough default)
    # Map x-range to world_width units. Adjust later for nice placement.
    global SCALE, OFFSET
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    SCALE = float(width) / float(xmax - xmin)   # simple guess; you will tune
    OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    _CACHE.update({
        "initialized": True,
        "frame_idx": 0,
        "centres": centres,
        "lut": lut,
        "img_size": img_size,
        "bg_sub": bg_sub,
        "thresholds": thresholds,
        "kernel": kernel,
        "bounds": bounds,
    })

def set_voxel_positions(width, height, depth):
    # Called from OpenGL when you press G.
    # Returns positions/colors for cube.set_multiple_positions(...)

    if not _CACHE["initialized"]:
        _init_voxel_system(width, height, depth)

    frame_idx = _CACHE["frame_idx"]

    # 1) build masks for this frame
    masks = get_mask_for_frame(frame_idx, _CACHE["bg_sub"], _CACHE["thresholds"], _CACHE["kernel"])

    # 2) (optional but recommended) dilate masks slightly for tolerance
    # helps small calibration/segmentation errors
    for cam in range(1, 5):
        if masks[cam] is not None:
            masks[cam] = cv.dilate(masks[cam], cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), iterations=1)


    # 3) carve
    active = remove_voxels(_CACHE["lut"], masks, rule="AND")

    centres_flat = _CACHE["centres"].reshape(-1,3)
    pts = centres_flat[active]
    print("ACTIVE bbox min", pts.min(axis=0), "max", pts.max(axis=0))
    # ---- DEBUG: check consistency with each camera mask ----
    for cam in range(1,5):
        uv = _CACHE["lut"][f"cam{cam}"]["uv"]
        valid = _CACHE["lut"][f"cam{cam}"]["valid"]

        idxs = np.where(active & valid)[0]
        if len(idxs) == 0:
            print(f"cam{cam}: no active voxels")
            continue

        inside = np.mean(masks[cam][uv[idxs,1], uv[idxs,0]] > 0)
        print(f"cam{cam} active-inside-mask:", inside)

    # 4) render lists
    positions, colors = render_lists(_CACHE["centres"], active)

    # 5) advance frame for next press (or keep fixed if you want)
    _CACHE["frame_idx"] += 1
    print("frame", frame_idx, "active:", int(active.sum()), "positions:", len(positions))
    centres_flat = _CACHE["centres"].reshape(-1, 3)
    pts = centres_flat[active]
    print("calib xyz min", pts.min(axis=0), "max", pts.max(axis=0))
    return positions, colors

if __name__ == "__main__":

    # -----------------------------
    # CONFIG FOR DEBUGGING
    # -----------------------------
    frame_idx = 0
    k = 5
    n_train = 200
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    bounds = (-1000, 1000, 0, 2000, -1000, 1000)  # (xmin,xmax,ymin,ymax,zmin,zmax)
    vox_w, vox_h, vox_d = 200,200,200

    # -----------------------------
    # 1) Compute image sizes once
    # -----------------------------
    img_size = {}
    for cam in range(1, 5):
        cap = cv.VideoCapture(f"data/cam{cam}/video.avi")
        ret, frame0 = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read first frame for cam{cam}")
        H, W = frame0.shape[:2]
        img_size[cam] = (W, H)

    # -----------------------------
    # 2) Build voxel centres + LUT once
    # -----------------------------
    centres, _ = find_voxel_centres(vox_w, vox_h, vox_d, bounds)
    lut = construct_lookup(centres, params, img_size)
    centres_flat = centres.reshape(-1, 3)
    n_vox = centres_flat.shape[0]

    # -----------------------------
    # 3) Train background models + thresholds once
    # -----------------------------
    bg_sub = {}
    thresholds = {}
    for cam in range(1, 5):
        bg_sub[cam] = background_detection(f"data/cam{cam}/background.avi", k=k, n_train=n_train)

        candidates = [
            f"data/cam{cam}/manual.png",
            f"data/cam{cam}/mask.png",
            f"data/cam{cam}/segmentation.png",
            f"data/cam{cam}/foreground.png",
        ]
        img_path = next((p for p in candidates if os.path.exists(p)), None)

        if img_path is None:
            thresholds[cam] = (10, 25, 25)
            print(f"cam{cam}: no manual mask found -> using default thresholds {thresholds[cam]}")
        else:
            thresholds[cam], best_score = get_thresholds(
                f"data/cam{cam}/video.avi",
                img_path,
                bg_sub[cam],
                frame_idx=0
            )
            print(f"cam{cam} thresholds={thresholds[cam]} score={best_score:.1f}")

    # -----------------------------
    # 4) Print extrinsics sanity ONCE
    # -----------------------------
    print("\n=== EXTRINSICS SANITY ===")
    for cam in range(1, 5):
        rvec = params[f"cam{cam}"][2]
        tvec = params[f"cam{cam}"][3]
        R, _ = cv.Rodrigues(rvec)

        C = (-R.T @ tvec).reshape(3)                  # camera center in world coords
        forward = (R.T @ np.array([0, 0, 1.0])).reshape(3)  # camera +Z in world
        forward = forward / (np.linalg.norm(forward) + 1e-9)

        print(f"cam{cam}: C(world)={C}  forward(world)={forward}")

    # -----------------------------
    # helper: draw world axes (quick extrinsic check)
    # -----------------------------
    def draw_world_axes(frame_bgr, cam, origin=(0, 0, 0), axis_len=300):
        """
        Projects 3D axes from world origin into the camera image.
        If extrinsics are wrong, axes will be off-screen or nonsensical.
        """
        camera_matrix, dist, rvec, tvec = params[f"cam{cam}"]
        O = np.array(origin, dtype=np.float32)
        pts = np.array([
            O,
            O + np.array([axis_len, 0, 0], dtype=np.float32),  # +X
            O + np.array([0, axis_len, 0], dtype=np.float32),  # +Y
            O + np.array([0, 0, axis_len], dtype=np.float32),  # +Z
        ], dtype=np.float32).reshape(-1, 1, 3)

        imgpts, _ = cv.projectPoints(pts, rvec, tvec, camera_matrix, dist)
        imgpts = np.rint(imgpts).astype(int).reshape(-1, 2)

        # only draw if origin projects into image
        h, w = frame_bgr.shape[:2]
        ox, oy = imgpts[0]
        if not (0 <= ox < w and 0 <= oy < h):
            return frame_bgr  # origin not visible; still a useful sign

        # draw axes (BGR): X=red, Y=green, Z=blue
        xpt = tuple(imgpts[1]); ypt = tuple(imgpts[2]); zpt = tuple(imgpts[3])
        cv.line(frame_bgr, (ox, oy), xpt, (0, 0, 255), 2)
        cv.line(frame_bgr, (ox, oy), ypt, (0, 255, 0), 2)
        cv.line(frame_bgr, (ox, oy), zpt, (255, 0, 0), 2)
        cv.circle(frame_bgr, (ox, oy), 4, (255, 255, 255), -1)
        return frame_bgr

    # -----------------------------
    # 5) Interactive frame stepping
    # -----------------------------
    while True:
        print(f"\n=== FRAME {frame_idx} === (a=prev, d=next, q/esc=quit)")

        # a) masks for this frame
        masks = get_mask_for_frame(frame_idx, bg_sub, thresholds, kernel)

        # mask stats
        for cam in range(1, 5):
            m = masks[cam]
            if m is None:
                print(f"cam{cam}: mask=None")
                continue
            fg_ratio = float(np.mean(m > 0))
            fg_area = int(np.count_nonzero(m))
            print(f"cam{cam}: fg_ratio={fg_ratio:.4f} fg_area={fg_area}")

        # b) carve
        active = remove_voxels(lut, masks, rule="AND")
        active_count = int(active.sum())
        print("Active voxels:", active_count, "out of", n_vox)

        # c) head-band diagnostic (tune these numbers)
        Y = centres_flat[:, 1]
        head_band = (Y >= 1000) & (Y <= 1500)
        print("head band active fraction:", float(np.mean(active[head_band])))

        # d) THE KEY TEST: active voxels must be inside mask in EVERY camera
        for cam in range(1, 5):
            uv = lut[f"cam{cam}"]["uv"]
            valid = lut[f"cam{cam}"]["valid"]
            idxs = np.where(active & valid)[0]
            if len(idxs) == 0:
                print(f"cam{cam}: active-inside-mask: n=0")
                continue
            inside = float(np.mean(masks[cam][uv[idxs, 1], uv[idxs, 0]] > 0))
            print(f"cam{cam}: active-inside-mask={inside:.3f} (n={len(idxs)})")

        # e) show windows for each cam:
        #    - frame
        #    - mask
        #    - overlay (mask on frame)
        #    - projection (active voxels reprojected)
        for cam in range(1, 5):
            cap = cv.VideoCapture(f"data/cam{cam}/video.avi")
            cap.set(cv.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(f"cam{cam}: could not read frame {frame_idx}")
                continue

            mask = masks[cam]
            if mask is None:
                continue

            overlay = frame.copy()
            overlay[mask > 0] = (0, 255, 0)

            # draw axes to sanity-check extrinsics
            overlay_axes = draw_world_axes(overlay.copy(), cam, origin=(0, 0, 0), axis_len=300)

            # project active voxels
            proj = frame.copy()
            uv = lut[f"cam{cam}"]["uv"]
            valid = lut[f"cam{cam}"]["valid"]
            idxs = np.where(valid & active)[0]

            if len(idxs) > 0:
                sample = np.random.choice(idxs, size=min(8000, len(idxs)), replace=False)
                for p in sample:
                    u, v = uv[p]
                    cv.circle(proj, (int(u), int(v)), 2, (0, 255, 255), -1)

            cv.imshow(f"cam{cam} frame", frame)
            cv.imshow(f"cam{cam} mask", mask)
            cv.imshow(f"cam{cam} overlay(mask)", overlay)
            cv.imshow(f"cam{cam} overlay+axes", overlay_axes)
            cv.imshow(f"cam{cam} voxel projections", proj)

        key = cv.waitKey(0) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('d'):
            frame_idx += 1
        elif key == ord('a'):
            frame_idx = max(0, frame_idx - 1)

        cv.destroyAllWindows()

    cv.destroyAllWindows()



"""
#################################################################################################################################
"""
"""
    Choice task 4: Implementing the surface mesh
"""

def voxels_to_volume(voxels, voxel_size= block_size, padding=1):
    """
    world-coordinate voxels -> 3d grid
    
    ret:
        volume 3d numpy arr
        world coordinate origin in volume
    """
    # Compute bounds
    min_coord = voxels.min(axis=0) - padding * voxel_size
    max_coord = voxels.max(axis=0) + padding * voxel_size
    dims = np.ceil((max_coord - min_coord) / voxel_size).astype(int) + 1
    volume = np.zeros(dims, dtype=np.uint8)
    
    # Map voxel positions to indices
    idxs = np.floor((voxels - min_coord) / voxel_size).astype(int)
    volume[idxs[:,0], idxs[:,1], idxs[:,2]] = 1
    
    return volume, min_coord

def volume_to_mesh(volume, voxel_size=2.0, origin=np.array([0,0,0])):
    """
    3d grid -> vertices + faces (using MarchingCubes)
    """
    verts, faces, normals, values = measure.marching_cubes(volume, level=0.5)
    
    # Transform from voxel indices to world coordinates
    verts_world = verts * voxel_size + origin
    
    return verts_world, faces

def save_mesh_obj(filename, verts, faces):
    """
    Save mesh as a wavefront obj
    """
    with open(filename, "w") as f:
        f.write("# OBJ file generated from voxel marching cubes\n")

        # vertices
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # faces
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

positions, colors = set_voxel_positions()
voxels_np = np.array(positions)

# voxels -> 3d volume -> mesh -> obj
volume, origin = voxels_to_volume(voxels_np, voxel_size=2.0, padding=1)
verts, faces = volume_to_mesh(volume, voxel_size=2.0, origin=origin)
save_mesh_obj("mesh.obj", verts, faces)