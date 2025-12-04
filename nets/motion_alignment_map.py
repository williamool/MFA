import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk


def butterworth_highpass(img, d0=30, n=2):
    """
    对单帧图像进行频域Butterworth高通滤波
    
    Args:
        img: 单帧图像 (H, W) - 灰度图
        d0: 截止频率（默认30，增大可保留更多低频信息，减小可过滤更多低频）
        n: 滤波器阶数（默认2，增大可使过渡更陡峭）
    
    Returns:
        滤波后的图像 (H, W)
    """
    h, w = img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    du, dv = u - w//2, v - h//2
    D = np.sqrt(du**2 + dv**2)
    
    # 计算Butterworth高通滤波器
    H = 1 / (1 + (d0 / (D + 1e-5))**(2 * n))
    
    # 频域变换
    img_dft = np.fft.fftshift(np.fft.fft2(img))
    img_hp = np.real(np.fft.ifft2(np.fft.ifftshift(img_dft * H)))
    
    return img_hp


def compute_saliency(img):
    """
    计算Laplacian熵显著性图
    
    Args:
        img: 输入图像 (H, W) - 灰度图
    
    Returns:
        显著性图 (H, W)
    """
    # 确保图像是uint8类型
    if img.dtype != np.uint8:
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    
    # 计算Laplacian算子（二阶导数，检测边缘）
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    
    # 计算熵（使用局部窗口）
    entropy_map = entropy(np.abs(laplacian).astype(np.uint8), disk(3))
    
    return entropy_map


def motion_compensate(frame1, frame2, 
                      motion_distance_threshold=50,
                      min_tracking_points=15,
                      ransac_threshold=1.0,
                      klt_epsilon=0.003,
                      klt_max_iter=30):
    """
    基于网格的KLT光流跟踪进行运动补偿
    
    Args:
        frame1: 前一帧图像（灰度图）
        frame2: 当前帧图像（灰度图）
        motion_distance_threshold: 运动距离阈值，超过此值的点会被过滤（默认50）
        min_tracking_points: 最小跟踪点数量，少于此时使用单位矩阵（默认15）
        ransac_threshold: RANSAC重投影误差阈值（默认1.0）
        klt_epsilon: KLT光流精度阈值（默认0.003）
        klt_max_iter: KLT光流最大迭代次数（默认30）
    
    Returns:
        compensated: 运动补偿后的图像
        mask: 掩膜
        avg_dst: 平均运动距离
        motion_x: 平均x方向运动
        motion_y: 平均y方向运动
        homography_matrix: 单应性矩阵
    """
    # KLT光流跟踪参数
    lk_params = dict(winSize=(15, 15), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, klt_max_iter, klt_epsilon))

    width = frame2.shape[1]
    height = frame2.shape[0]
    scale = 4  # 放大倍数，提高跟踪精度

    # 将图像放大以提高跟踪精度
    new_width = int(width * scale)
    new_height = int(height * scale)
    frame1_grid = cv2.resize(frame1, (new_width, new_height), dst=None, interpolation=cv2.INTER_CUBIC)
    frame2_grid = cv2.resize(frame2, (new_width, new_height), dst=None, interpolation=cv2.INTER_CUBIC)

    width_grid = frame2_grid.shape[1]
    height_grid = frame2_grid.shape[0]
    
    # 生成网格点
    gridSizeW = 32 * 4
    gridSizeH = 24 * 4
    p1 = []
    grid_numW = int(width_grid / gridSizeW - 1)
    grid_numH = int(height_grid / gridSizeH - 1)
    
    for i in range(grid_numW):
        for j in range(grid_numH):
            point = (np.float32(i * gridSizeW + gridSizeW / 2.0), 
                     np.float32(j * gridSizeH + gridSizeH / 2.0))
            p1.append(point)

    p1 = np.array(p1)
    pts_num = grid_numW * grid_numH
    pts_prev = p1.reshape(pts_num, 1, 2)

    # KLT光流跟踪
    pts_cur, st, err = cv2.calcOpticalFlowPyrLK(frame1_grid, frame2_grid, pts_prev, None, **lk_params)

    # 选择有效的跟踪点
    good_new = pts_cur[st == 1]  # 当前帧中的跟踪点
    good_old = pts_prev[st == 1]  # 前一帧中的跟踪点

    # 计算运动距离和方向，过滤异常点
    motion_distance = []
    translate_x = []
    translate_y = []
    
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        motion_distance0 = np.sqrt((a - c) * (a - c) + (b - d) * (b - d))

        # 过滤运动距离过大的点（可能是噪声）
        if motion_distance_threshold is not None and motion_distance0 > motion_distance_threshold:
            continue

        translate_x0 = a - c
        translate_y0 = b - d

        motion_distance.append(motion_distance0)
        translate_x.append(translate_x0)
        translate_y.append(translate_y0)
    
    motion_dist = np.array(motion_distance)
    motion_x = np.mean(np.array(translate_x)) if len(translate_x) > 0 else 0
    motion_y = np.mean(np.array(translate_y)) if len(translate_y) > 0 else 0
    avg_dst = np.mean(motion_dist) if len(motion_dist) > 0 else 0

    # 计算单应性矩阵
    if len(good_old) < min_tracking_points:
        # 如果跟踪点太少，使用单位矩阵（几乎不变换）
        homography_matrix = np.array([[0.999, 0, 0], [0, 0.999, 0], [0, 0, 1]])
    else:
        # 使用RANSAC算法计算单应性矩阵
        homography_matrix, status = cv2.findHomography(good_new, good_old, cv2.RANSAC, ransac_threshold)

    # 根据变换矩阵计算变换之后的图像（运动补偿）
    compensated = cv2.warpPerspective(frame1, homography_matrix, (width, height), 
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # 计算掩膜（用于标记变换后的有效区域）
    vertex = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32).reshape(-1, 1, 2)
    homo_inv = np.linalg.inv(homography_matrix)
    vertex_trans = cv2.perspectiveTransform(vertex, homo_inv)
    vertex_transformed = np.array(vertex_trans, dtype=np.int32).reshape(1, 4, 2)
    im = np.zeros(frame1.shape[:2], dtype='uint8')
    cv2.polylines(im, vertex_transformed, 1, 255)
    cv2.fillPoly(im, vertex_transformed, 255)
    mask = 255 - im

    return compensated, mask, avg_dst, motion_x, motion_y, homography_matrix


def background_suppression(energy_map, threshold=0.1):
    """
    背景抑制，减少噪声干扰
    
    Args:
        energy_map: 能量图 (H, W)，值范围[0, 1]
        threshold: 阈值（默认0.1，小于此值的区域被抑制）
    
    Returns:
        抑制后的能量图 (H, W)，uint8格式，值范围[0, 255]
    """
    # 阈值处理
    energy_map = energy_map.copy()
    energy_map[energy_map < threshold] = 0
    
    # 中值滤波去噪
    energy_map_uint8 = (energy_map * 255).astype(np.uint8)
    suppressed = cv2.medianBlur(energy_map_uint8, 3)
    
    return suppressed


def generate_motion_alignment_map(frame1, frame2,
                                  motion_distance_threshold=50,
                                  min_tracking_points=15,
                                  ransac_threshold=1.0,
                                  klt_epsilon=0.003,
                                  klt_max_iter=30,
                                  suppression_threshold=0.1,
                                  use_highpass=True,
                                  highpass_d0=30,
                                  highpass_n=2):
    """
    生成Motion Difference Map（两帧输入）
    
    流程：输入两帧 -> 高通滤波（可选）-> 光流跟踪计算差分图 -> 输出
    
    Args:
        frame1: 前一帧图像，可以是numpy数组 (H, W) 或 (H, W, 3)，或图像路径
        frame2: 当前帧图像，可以是numpy数组 (H, W) 或 (H, W, 3)，或图像路径
        motion_distance_threshold: 运动距离阈值，超过此值的点会被过滤（默认50）
                                   - 对于微弱运动：可增大到100-200或设为None保留所有点
                                   - 对于强运动：可减小到20-30过滤噪声
        min_tracking_points: 最小跟踪点数量（默认15，减小可降低要求）
        ransac_threshold: RANSAC重投影误差阈值（默认1.0）
                         - 对于微弱运动：可减小到1.0-2.0提高精度
        klt_epsilon: KLT光流精度阈值（默认0.003）
                    - 对于微弱运动：可减小到0.001-0.002提高敏感度
        klt_max_iter: KLT光流最大迭代次数（默认30）
        suppression_threshold: 背景抑制阈值（默认0.1，已注释，不再使用）
                              - 小于此值的区域被抑制
        use_highpass: 是否使用高频滤波（默认True）
                     - True: 启用Butterworth高通滤波，突出高频信息（边缘、细节）
                     - False: 不使用高频滤波
        highpass_d0: 高通滤波截止频率（默认30）
                    - 增大：保留更多低频信息
                    - 减小：过滤更多低频，只保留高频（边缘、细节）
        highpass_n: 高通滤波器阶数（默认2）
                   - 增大：过渡更陡峭，滤波效果更明显
    
    Returns:
        motion_diff_map: Motion Difference Map（numpy数组，uint8格式，值范围[0, 255]）
    """
    # 读取图像（如果输入是路径）
    if isinstance(frame1, str):
        frame1 = cv2.imread(frame1)
        if frame1 is None:
            raise ValueError(f"无法读取图像: {frame1}")
    
    if isinstance(frame2, str):
        frame2 = cv2.imread(frame2)
        if frame2 is None:
            raise ValueError(f"无法读取图像: {frame2}")
    
    # 确保所有图像尺寸相同
    target_shape = frame1.shape[:2]  # (H, W)
    if frame2.shape[:2] != target_shape:
        frame2 = cv2.resize(frame2, (target_shape[1], target_shape[0]))
    
    # 转换为灰度图
    if len(frame1.shape) == 3:
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    else:
        frame1_gray = frame1.copy()
    
    if len(frame2.shape) == 3:
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        frame2_gray = frame2.copy()
    
    # 高通滤波（可选）- 对所有帧进行滤波
    if use_highpass:
        # 应用高频滤波
        frame1_gray = butterworth_highpass(frame1_gray, d0=highpass_d0, n=highpass_n)
        frame2_gray = butterworth_highpass(frame2_gray, d0=highpass_d0, n=highpass_n)
        
        # 归一化到0-255范围（频域滤波可能产生负值或超出范围的值）
        def normalize_frame(frame):
            frame_min, frame_max = frame.min(), frame.max()
            if frame_max > frame_min:
                return ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
            else:
                return np.clip(frame, 0, 255).astype(np.uint8)
        
        frame1_gray = normalize_frame(frame1_gray)
        frame2_gray = normalize_frame(frame2_gray)
    
    # 步骤1: 光流跟踪计算差分图
    img_compensate, mask, avg_dist, motion_x, motion_y, homo_matrix = motion_compensate(
        frame1_gray, frame2_gray,
        motion_distance_threshold=motion_distance_threshold,
        min_tracking_points=min_tracking_points,
        ransac_threshold=ransac_threshold,
        klt_epsilon=klt_epsilon,
        klt_max_iter=klt_max_iter)
    
    # 计算Motion Difference Map
    frame_diff = cv2.absdiff(frame2_gray, img_compensate)
    
    # 步骤2: 显著性计算（已注释）
    # saliency_map = compute_saliency(frame_diff)
    # 
    # # 归一化显著性图到0-255范围
    # saliency_min, saliency_max = saliency_map.min(), saliency_map.max()
    # if saliency_max > saliency_min:
    #     saliency_map = ((saliency_map - saliency_min) / (saliency_max - saliency_min) * 255).astype(np.uint8)
    # else:
    #     saliency_map = np.clip(saliency_map, 0, 255).astype(np.uint8)
    # 
    # # 归一化到[0, 1]范围用于背景抑制
    # saliency_normalized = saliency_map.astype(np.float32) / 255.0
    
    # 步骤3: 背景抑制（已注释）
    # motion_alignment_map = background_suppression(saliency_normalized, threshold=suppression_threshold)
    
    # 直接返回Motion Difference Map
    return frame_diff


def generate_motion_alignment_map_batch(frame1_batch, frame2_batch,
                                        motion_distance_threshold=50,
                                        min_tracking_points=15,
                                        ransac_threshold=1.0,
                                        klt_epsilon=0.003,
                                        klt_max_iter=30,
                                        use_highpass=True,
                                        highpass_d0=30,
                                        highpass_n=2):
    """
    批量生成Motion Difference Map（支持tensor输入）
    
    Args:
        frame1_batch: 前一帧图像批次，numpy数组 (B, H, W) 或 (B, H, W, 3)，或torch.Tensor (B, C, H, W)
        frame2_batch: 当前帧图像批次，numpy数组 (B, H, W) 或 (B, H, W, 3)，或torch.Tensor (B, C, H, W)
        其他参数同 generate_motion_alignment_map
    
    Returns:
        motion_diff_maps: Motion Difference Map批次，numpy数组 (B, H, W)，uint8格式
    """
    import torch
    
    # 处理tensor输入
    if isinstance(frame1_batch, torch.Tensor):
        # 转换为numpy并处理维度
        if frame1_batch.dim() == 4:  # (B, C, H, W)
            # 转换为 (B, H, W, C) 然后转换为灰度或保持RGB
            frame1_np = frame1_batch.permute(0, 2, 3, 1).cpu().numpy()
            # 如果值在[0,1]范围，转换为[0,255]
            if frame1_np.max() <= 1.0:
                frame1_np = (frame1_np * 255).astype(np.uint8)
            else:
                frame1_np = frame1_np.astype(np.uint8)
        else:
            frame1_np = frame1_batch.cpu().numpy()
    else:
        frame1_np = frame1_batch
    
    if isinstance(frame2_batch, torch.Tensor):
        if frame2_batch.dim() == 4:  # (B, C, H, W)
            frame2_np = frame2_batch.permute(0, 2, 3, 1).cpu().numpy()
            if frame2_np.max() <= 1.0:
                frame2_np = (frame2_np * 255).astype(np.uint8)
            else:
                frame2_np = frame2_np.astype(np.uint8)
        else:
            frame2_np = frame2_batch.cpu().numpy()
    else:
        frame2_np = frame2_batch
    
    batch_size = frame1_np.shape[0]
    motion_diff_maps = []
    
    # 对每个batch计算运动差分图
    for i in range(batch_size):
        frame1 = frame1_np[i]
        frame2 = frame2_np[i]
        
        # 调用单帧处理函数
        motion_diff = generate_motion_alignment_map(
            frame1=frame1,
            frame2=frame2,
            motion_distance_threshold=motion_distance_threshold,
            min_tracking_points=min_tracking_points,
            ransac_threshold=ransac_threshold,
            klt_epsilon=klt_epsilon,
            klt_max_iter=klt_max_iter,
            suppression_threshold=0.1,
            use_highpass=use_highpass,
            highpass_d0=highpass_d0,
            highpass_n=highpass_n
        )
        motion_diff_maps.append(motion_diff)
    
    return np.stack(motion_diff_maps, axis=0)  # (B, H, W)


if __name__ == "__main__":
    # 测试代码
    img1_path = r"D:\Github\ITSDT\images\21\51.bmp"  # 帧 t-1
    img2_path = r"D:\Github\ITSDT\images\21\52.bmp"  # 帧 t
    
    try:
        motion_diff_map = generate_motion_alignment_map(
            frame1=img1_path,
            frame2=img2_path,
            # 微弱运动目标优化参数（可根据实际情况调整）
            motion_distance_threshold=None,  # None表示保留所有点，或设为100-200
            min_tracking_points=10,          # 降低跟踪点数量要求
            ransac_threshold=1.0,            # 减小以提高精度
            klt_epsilon=0.01,               # 减小以提高对微弱运动的敏感度
            klt_max_iter=10,                 # 迭代次数
            suppression_threshold=0.2,       # 背景抑制阈值（已注释，不再使用）
            # 高频滤波参数
            use_highpass=True,               # 启用高频滤波
            highpass_d0=50,                  # 截止频率（可调整：20-50）
            highpass_n=2                     # 滤波器阶数（可调整：1-4）
        )
        
        print(f"成功生成Motion Difference Map!")
        print(f"输出图像尺寸: {motion_diff_map.shape}")
        print(f"像素值范围: [{motion_diff_map.min()}, {motion_diff_map.max()}]")
        
        # 以热力图形式可视化展示
        plt.figure(figsize=(10, 8))
        plt.imshow(motion_diff_map, cmap='hot', interpolation='bilinear')
        plt.colorbar(label='Motion Intensity')
        plt.title('Motion Difference Map (Heatmap)', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

