from turtle import width

import glm
import cv2 as cv
import random
import numpy as np
import os
import glob
import shutil


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
"""
Referencing image: https://uu.brightspace.com/content/enforced/44949-BETA--2025--3-GS--INFOMCV--V/csfiles/home_dir/courses/BETA-2023-3-GS-INFOMCV-V/BETA-2023-3-GS-INFOMCV-V/flow.png

-   Now we have a function to get a set of voxels visible in all cameras (active voxels).
    (TODO: for now it loops through 4 sample frames but it should be adapted to loop through all frames in the video)

-   NOTE: when choosing the parameters to call the function, choose them wisely so that the voxels are positioned nicely (not too small/big, etc.)
    This was something Mr.Boobs stressed on in the lecture

-   The output of the function is a set of voxels that are active in all camera views, which can be used for 3D reconstruction or visualization.

-   TODO: we need to give the 3d array of turned on voxels to the visualization code

"""