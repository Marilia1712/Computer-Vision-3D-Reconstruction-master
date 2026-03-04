from turtle import width

import glm
import cv2 as cv
import random
import numpy as np
import os
import glob
import shutil


block_size = 1.0
pattern_size = (8, 6)
EDGE_SIZE = 115 # size of the square edge (in mm)

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
   
def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data, colors = [], []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if random.randint(0, 1000) < 5:
                    data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
                    colors.append([x / width, z / depth, y / height])
    return data, colors

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

def signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

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

"""
    Exercise 1
"""

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

def reproj_error(objp, imgp, rvec, tvec, K, dist):
    proj, _ = cv.projectPoints(objp, rvec, tvec, K, dist)
    return np.mean(np.linalg.norm(proj.reshape(-1,2) - imgp.reshape(-1,2), axis=1))

def save_camera_config_xml(filename, K, dist, rvec=None, tvec=None):
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)

    fs.write("CameraMatrix", K.astype(np.float32))
    fs.write("DistortionCoeffs", dist.astype(np.float32))

    if rvec is not None:
        fs.write("rvec", rvec.astype(np.float32))
    if tvec is not None:
        fs.write("tvec", tvec.astype(np.float32))

    fs.release()

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

            if ret == True:
                print(f"\nFrame {i}: chessboard detected")

                corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

                h, w = gray.shape

                # ==============================
                # 1. Coverage check
                # ==============================
                xs = corners2[:,0,0]
                ys = corners2[:,0,1]

                cov_x = (xs.max() - xs.min()) / w
                cov_y = (ys.max() - ys.min()) / h

                print(f"  coverage x={cov_x:.2f} y={cov_y:.2f}")

                if cov_x < 0.2 or cov_y < 0.2:
                    print("  ❌ Rejected: poor spatial coverage")
                    continue

                """
                # ==============================
                # 2. Corner stability check
                # ==============================
                movement = np.mean(np.linalg.norm(corners2 - corners, axis=2))
                print(f"  refinement movement = {movement:.3f}px")

                if movement > 0.5:
                    print("  ❌ Rejected: unstable/blurred corners")
                    continue
                """

                # ==============================
                # 3. Board area check
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
                
            else:
                #cv.imwrite(f'{dest_manual}/frame_{i}_{video_path[11:-5]}.jpg', frame)
            
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

                # --------------- Enforce order -----------------#

                src = np.array(state['2dpoints'], dtype = np.float32)
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


# #1.2 estrinsic parameters 
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
    Exercise 2
"""

def background_detection(video_path):
    back = cv.VideoCapture(video_path)
    n_frames = back.get(cv.CAP_PROP_FRAME_COUNT)
    print("Number of frames: ", n_frames)

    #take half of the frames
    frames_indeces = np.linspace(0, n_frames, int(n_frames//2), dtype = np.int32)
    print("Frames considered: ", len(frames_indeces))

    frames_back = []

    #Preprocess all frames
    for index in frames_indeces:
        back.set(cv.CAP_PROP_POS_FRAMES, index)
        ret, frame = back.read()
        if not ret:
            continue
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frames_back.append(frame_hsv)

    all_frames_back = np.stack(frames_back, axis = 0)

    #Take median
    background = np.median(all_frames_back, axis = 0)
    background = background.astype(np.uint8)
    # cv.imshow("Background", background)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return background


def foreground_mask_at_frame(video_path, background, kernel, frame_idx=0, k=3,
                             extra_dilate=2, extra_close=2):
    """
    Compute ONE foreground mask at a chosen frame index.
    Also applies extra morphology to "turn on" missing pixels (holes, thin regions).
    """
    cap = cv.VideoCapture(video_path)
    cap.set(cv.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    fore_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    hue_diff = cv.absdiff(fore_hsv[:, :, 0], background[:, :, 0])
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)
    sat_diff = cv.absdiff(fore_hsv[:, :, 1], background[:, :, 1])
    val_diff = cv.absdiff(fore_hsv[:, :, 2], background[:, :, 2])

    mean_h, std_h = float(np.mean(hue_diff)), float(np.std(hue_diff))
    mean_s, std_s = float(np.mean(sat_diff)), float(np.std(sat_diff))
    mean_v, std_v = float(np.mean(val_diff)), float(np.std(val_diff))

    thresh_hue = mean_h + k * std_h
    thresh_sat = mean_s + k * std_s
    thresh_val = mean_v + k * std_v

    _, mask_h = cv.threshold(hue_diff, thresh_hue, 255, cv.THRESH_BINARY)
    _, mask_s = cv.threshold(sat_diff, thresh_sat, 255, cv.THRESH_BINARY)
    _, mask_v = cv.threshold(val_diff, thresh_val, 255, cv.THRESH_BINARY)

    mask = cv.bitwise_or(mask_h, mask_s)
    mask = cv.bitwise_or(mask, mask_v)

    # Basic clean
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

    # ---- IMPORTANT: "turn on" missing pixels ----
    # Close fills holes; dilate fattens silhouette (helps voxel intersection)
    if extra_close > 0:
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=extra_close)
    if extra_dilate > 0:
        mask = cv.dilate(mask, kernel, iterations=extra_dilate)

    return mask

"""
#loop to get background subtraction for each camera

for i in range(1,5):
    video_path_back = f"data/cam{i}/background.avi"
    video_path_fore = f"data/cam{i}/video.avi"
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))

    background = background_detection(video_path_back)
    foreground = background_subtraction(video_path_fore, background, kernel)

"""

"""
Exercise 3
"""

# Get camera position (adaptation of code from Assingnment 1)

cam_pos = {}
cam_orient = {}
cam_R = {}

rvecs = {}
tvecs = {}
cam_matrix = {}
dist_coeff = {}
params = {}

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


def generate_voxel_grid():
    ranges = [(-50,50), (0,150), (-50,50)]
    voxel_dim = 2.0
    axes = [np.arange(lo, hi, voxel_dim, dtype=np.float32) for lo, hi in ranges]
    grid = np.meshgrid(*axes, indexing='ij')
    return np.stack([g.ravel() for g in grid], axis=1)

def create_lookup_table_fast(voxels, cam_params):
    """
    Robust LUT:
      - projects all voxels at once per camera
      - filters invalid uv (NaN/Inf/huge)
      - filters voxels behind the camera (z_cam <= 0)
    Returns: lut[cam] = (u, v, valid)
    """
    lut = {}
    objp = voxels.reshape(-1, 1, 3).astype(np.float32)

    for cam in range(1, 5):
        camera_matrix, dist, rvec, tvec = cam_params[f'cam{cam}']

        camera_matrix = camera_matrix.astype(np.float64)
        dist = dist.astype(np.float64)
        rvec = rvec.astype(np.float64)
        tvec = tvec.astype(np.float64)

        # ---- front-of-camera filter (camera coords z > 0)
        R, _ = cv.Rodrigues(rvec)
        Xc = (R @ voxels.T + tvec).T  # (N,3)
        valid_front = Xc[:, 2] > 1e-6

        # ---- project
        imgpts, _ = cv.projectPoints(objp, rvec, tvec, camera_matrix, dist)
        uv = imgpts.reshape(-1, 2)

        # ---- validity mask MUST be created before &= operations
        valid = np.isfinite(uv).all(axis=1)

        # remove extreme finite junk
        MAX_PIX = 1e7
        valid &= (np.abs(uv[:, 0]) < MAX_PIX) & (np.abs(uv[:, 1]) < MAX_PIX)

        # apply front-of-camera constraint
        valid &= valid_front

        # ---- fill u,v
        u = np.full(len(uv), -1, dtype=np.int32)
        v = np.full(len(uv), -1, dtype=np.int32)

        uv_valid = uv[valid].astype(np.float64)
        u[valid] = np.rint(uv_valid[:, 0]).astype(np.int32)
        v[valid] = np.rint(uv_valid[:, 1]).astype(np.int32)

        lut[cam] = (u, v, valid)

    return lut
    
def build_bg_gaussian(video_path, max_frames=80, sample_every=2, eps=1e-6):
    """
    Build per-pixel Gaussian model on BGR (or HSV if you want).
    Returns: mu (H,W,3) float32, var (H,W,3) float32
    """
    cap = cv.VideoCapture(video_path)
    n = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    frames = []
    idxs = list(range(0, n, sample_every))
    if len(idxs) > max_frames:
        idxs = np.linspace(0, n-1, max_frames, dtype=np.int32).tolist()

    for i in idxs:
        cap.set(cv.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame.astype(np.float32))

    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"No frames read from {video_path}")

    stack = np.stack(frames, axis=0)          # (T,H,W,3)
    mu = stack.mean(axis=0)                   # (H,W,3)
    var = stack.var(axis=0) + eps             # (H,W,3)
    return mu.astype(np.float32), var.astype(np.float32)

def fg_mask_gaussian_at_frame(video_path, mu, var, frame_idx=0,
                              tau=9.0,  # ~3-sigma per channel aggregated
                              kernel_size=3, close_it=2, open_it=1):
    """
    Foreground where normalized squared distance > tau.
    tau ~ 9 is a decent start for 3 channels.
    """
    cap = cv.VideoCapture(video_path)
    cap.set(cv.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    I = frame.astype(np.float32)
    diff2 = (I - mu) ** 2
    d = (diff2 / var).sum(axis=2)  # (H,W)

    mask = (d > tau).astype(np.uint8) * 255

    # morphology
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if open_it > 0:
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k, iterations=open_it)
    if close_it > 0:
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k, iterations=close_it)

    return mask

def set_voxel_positions(width, height, depth):
    """
    Fast voxel carving for ONE frame.
    Also fattens silhouettes to avoid missing voxels due to holes/thin masks.
    """

    if not hasattr(set_voxel_positions, "_cache"):
        set_voxel_positions._cache = {}
    cache = set_voxel_positions._cache

    # ---- Choose which frame you want to reconstruct
    f = 0  # change this if you want a different frame

    # ---- Backgrounds cached (compute once)
    if "bg_model" not in cache:
        bg_model = {}
        for cam in range(1,5):
            bg_path = f"data/cam{cam}/background.avi"
            mu, var = build_bg_gaussian(bg_path, max_frames=80, sample_every=2)
            bg_model[cam] = (mu, var)
        cache["bg_model"] = bg_model
    bg_model = cache["bg_model"]

    # ---- ONE foreground mask per cam for this frame (cache per frame index)
    key_masks = ("masks", f)
    if key_masks not in cache:
        masks = {}

        for cam in range(1, 5):
            video_path_fore = f"data/cam{cam}/video.avi"

            # get gaussian background model
            mu, var = bg_model[cam]

            mask = fg_mask_gaussian_at_frame(
                video_path_fore,
                mu,
                var,
                frame_idx=f,
                tau=9.0,
                kernel_size=3,
                close_it=2,
                open_it=1
            )

            if mask is None:
                return [], []

            masks[cam] = mask

        cache[key_masks] = masks
    masks = cache[key_masks]
    for cam in (1,2,3,4):
        nz = int(np.count_nonzero(masks[cam]))
        print(f"cam{cam} mask nonzero: {nz} / {masks[cam].size}")

    # ---- Voxels cached
    if "voxels" not in cache:
        SCALE = 10.0
        voxels = generate_voxel_grid().astype(np.float32) * SCALE
        cache["voxels"] = voxels
    voxels = cache["voxels"]

    # ---- LUT cached (vectorized)
    if "lut_fast" not in cache:
        cache["lut_fast"] = create_lookup_table_fast(voxels, params)
    lut = cache["lut_fast"]

    for cam in (1,2,3,4):
        u, v, valid = lut[cam]
        mask = masks[cam]
        h, w = mask.shape[:2]
        inb = (u>=0) & (u<w) & (v>=0) & (v<h)
        print(f"cam{cam} in-bounds projections: {inb.mean()*100:.2f}%")

    # ---- Vectorized carving

    N = voxels.shape[0]
    votes = np.zeros(N, dtype=np.uint8)

    for cam in (1,2,3,4):
        u, v, valid = lut[cam]
        mask = masks[cam]
        h, w = mask.shape[:2]

        inb = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = np.where(inb)[0]
        if idx.size == 0:
            continue

        fg = mask[v[idx], u[idx]] != 0
        votes[idx[fg]] += 1

    K = 3  # require 3 of 4 cameras (try 2 if too few)
    alive = np.where(votes >= K)[0]
    print("survivors (vote):", alive.size)

    if alive.size == 0:
        return [], []
    kept_voxels = voxels[alive].astype(np.float32)

    # ---- visualization transform ----
    SCALE = 10.0
    voxel_dim = 2.0
    step_world = SCALE * voxel_dim   # world distance between neighbors

    # 1) center in X,Z only (keep height meaningful)
    center_xz = kept_voxels.mean(axis=0)
    center_xz[1] = 0.0
    kept_centered = kept_voxels - center_xz

    # 2) put the lowest voxel on the floor (no negative height)
    kept_centered[:, 1] -= kept_centered[:, 1].min()

    # 3) increase spacing so cubes don't merge visually
    SPACING = 2.0                 # <--- makes it less dense
    VIS_SCALE = SPACING / step_world
    kept_vis = kept_centered * VIS_SCALE

    # If your visualizer floor is at y=-1, optionally lift a bit:
    # kept_vis[:, 1] += 1.0

    positions = np.column_stack([kept_vis[:, 0], kept_vis[:, 1], kept_vis[:, 2]])
    positions = positions.astype(float).tolist()
    colors = [[1.0, 1.0, 1.0] for _ in positions]
    return positions, colors