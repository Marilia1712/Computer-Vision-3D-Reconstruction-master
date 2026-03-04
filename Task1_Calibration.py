

from turtle import width
import glm
import cv2 as cv
import numpy as np
import os
import glob
import shutil
from skimage import measure  # marching_cubes
import json

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
    Task 1 + Choice Task 7: (improved) calibration
"""

MIN_POINTS = (pattern_size[0]*pattern_size[1]) // 4

# functions for camera calibration
def calibrate_camera(cameraNumber, video_path, dest, dest_manual, frame_number = 50):

    #video_path = f'data/cam{cameraNumber}/intrinsics.avi'
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

            cv.imwrite(f'{dest}/frame_{i}_{video_path[10:-5]}.jpg', frame)

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (pattern_size[0],pattern_size[1]), None)

            # case 1: full, straighforward localization of corners
            if ret == True:
                print(f"\nFrame {i}: chessboard detected")

                corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

                h, w = gray.shape

                # ==============================
                # 1. Coverage check                 -> TODO: move this to underneath the three cases
                # ==============================
                xs = corners2[:,0,0]
                ys = corners2[:,0,1]

                cov_x = (xs.max() - xs.min()) / w
                cov_y = (ys.max() - ys.min()) / h

                print(f"  coverage x={cov_x:.2f} y={cov_y:.2f}")

                if cov_x < 0.2 or cov_y < 0.2:
                    print("  ❌ Rejected: poor spatial coverage")
                    continue

                # ==============================
                # 2. Board area check
                # ==============================
                
                area = cv.contourArea(corners2.reshape(-1,2))
                img_area = w * h
                area_ratio = area / img_area

                print(f"  board area ratio = {area_ratio:.3f}")

                if area_ratio < 0.0025:
                    print("  ❌ Rejected: board too small/far")
                    continue
                

                # ==============================
                # ACCEPT FRAME
                # ==============================
                print("  ✅ Accepted")

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

            # case 2: partial localization of corners -> Convex Hull technique (Choice task 1)
            elif corners is not None and len(corners) >= MIN_POINTS:

                pts = corners.reshape(-1,2).astype(np.float32)
                hull = cv.convexHull(pts)
                rect = cv.minAreaRect(hull)
                box  = cv.boxPoints(rect)
                src = np.array(box, dtype=np.float32)

                # enforce order
                ordered = order_points(src, pattern_size = pattern_size)
                ordered_ref = ordered.reshape(-1, 1, 2).astype(np.float32)
                #cv.cornerSubPix(gray, ordered_ref, (11,11), (-1,-1), criteria)
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

                # --- Visualization for convex hull fallback ---
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

                #dst = np.float32([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]]) # the corners
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
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)

    fs.write("CameraMatrix", K.astype(np.float32))
    fs.write("DistortionCoeffs", dist.astype(np.float32))

    if rvec is not None:
        fs.write("rvec", rvec.astype(np.float32))
    if tvec is not None:
        fs.write("tvec", tvec.astype(np.float32))

    fs.release()

# functions for improved camera calibration (choice task 7)
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

    # -------------------------
    # First calibration
    # -------------------------
    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
        objectPoints, imagePoints, image_size, None, None,
        flags=0, criteria=criteria
    )

    if verbose:
        print("\n=== Per-frame reprojection errors ===")

    good_obj = []
    good_img = []

    errors = []

    # -------------------------
    # Compute reprojection error per frame
    # -------------------------
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
            print(f"Frame {idx:02d} → error = {err:.3f}px")

        if err < threshold:
            good_obj.append(objp)
            good_img.append(imgp)
        else:
            if verbose:
                print("   ❌ rejected")

    # -------------------------
    # If nothing rejected, skip recalibration
    # -------------------------
    if len(good_obj) == len(objectPoints):
        if verbose:
            print("\nNo frames rejected.")
        return K, dist, rvecs, tvecs

    # -------------------------
    # Recalibrate with good frames only
    # -------------------------
    if verbose:
        print(f"\nRecalibrating with {len(good_obj)}/{len(objectPoints)} good frames...")

    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
        good_obj, good_img, image_size, None, None,
        flags=0, criteria=criteria
    )

    if verbose:
        print("Done.\n")

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