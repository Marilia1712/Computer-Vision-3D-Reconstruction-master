from turtle import width

import glm
import cv2 as cv
import random
import numpy as np
import os
import glob
import shutil

# ---- WORLD MAPPING CONFIG ----
# Your calibration extrinsics (rvec/tvec) are in the same units as the checkerboard square size
# used when calibrating (Assignment 1). Very often that is in *centimeters*, not millimeters.
# We'll infer scale using ||tvec|| magnitudes.
SCALE = 1.0
OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float32)
block_size = 1.0

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