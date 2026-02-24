import glm
import cv2 as cv
import random
import numpy as np
import os
import glob

block_size = 1.0



video_path = 'data/cam1/intrinsics.avi'
pattern_size = (8, 6)
EDGE_SIZE = 115 # size of the square edge (in mm)
auto_objectPoints = []
auto_imagePoints = []
manual_objectPoints = []
manual_imagePoints = []
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
top_left = (0,0)
top_right = (pattern_size[0] - 1, 0)
bottom_left = (0, pattern_size[1] - 1)
bottom_right = (pattern_size[0] - 1, pattern_size[1] - 1)

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors

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



board_object_points = create_object_points(pattern_size[0], pattern_size[1],EDGE_SIZE)


frame_number = 10
cap = cv.VideoCapture(video_path)
i = 0

while i < frame_number:
    ret_, frame = cap.read()
    

    #..
    i += 1
    #_, rvec, tvec = cv.solvePnP(board_object_points, corners, cameraMatrix, distCoeffs)
    #img = cv.imread(frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (pattern_size[0],pattern_size[1]), None)

    if ret == True:
        print("Chessboard corners found in frame {}.".format(i))

        auto_objectPoints.append(board_object_points)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        auto_imagePoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(frame, (pattern_size[0],pattern_size[1]), corners2, ret)
        cv.imshow('img', frame)
        cv.waitKey(500)
        

    else:
        cv.imwrite('./manual_images/frame_{}.jpg'.format(i), frame)

    manual_images = glob.glob('./manual_images/*.jpg')

    for fname in manual_images:
    
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

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
        manual_imagePoints.append(img_points)
        manual_objectPoints.append(board_object_points.copy())

        for pt in img_grid:
            x, y = pt[0]
            cv.circle(state['interpolation'], (int(x), int(y)), 5, (255, 0, 0), -1)

        cv.imshow('Image', state["interpolation"])
        new_path = os.path.join("new_images", os.path.basename(fname))
        cv.imwrite(new_path, state["interpolation"])

        cv.waitKey(0)
        cv.destroyAllWindows()
