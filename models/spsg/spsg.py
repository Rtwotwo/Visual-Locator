import cv2
import numpy as np
import torch
from .superpoint import SuperPoint
from .superglue import SuperGlue 

def superpoint_match_pro(img1_path, img2_path, device='cuda', max_keypoints=-1, distance_ratio=0.85, ransac_threshold=2.0):
    image0 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    image1 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if image0 is None or image1 is None:
        raise FileNotFoundError("图像未找到，请检查路径")
    config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": max_keypoints,
        "remove_borders": 4,
        "path": "weights/superpoint_v1.pth",
        "cuda": (device == 'cuda')}
    model = SuperPoint(config).to(device)
    model.eval()
    def preprocess(img):
        img = cv2.equalizeHist(img)  # 增强对比度
        img_tensor = torch.from_numpy(img / 255.).float()[None, None].to(device)
        return img_tensor
    with torch.no_grad():
        pred0 = model({'image': preprocess(image0)})
        pred1 = model({'image': preprocess(image1)})
    
    keypoints0 = pred0["keypoints"][0].cpu().numpy()
    keypoints1 = pred1["keypoints"][0].cpu().numpy()
    
    descriptors0 = pred0["descriptors"][0].cpu().numpy().astype(np.float32).T
    descriptors1 = pred1["descriptors"][0].cpu().numpy().astype(np.float32).T
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors0, descriptors1, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < distance_ratio * n.distance:
            good_matches.append(m)
    
    # RANSAC 剔除外点
    inliers_matches = []
    if len(good_matches) >= 4:
        pts0 = np.float32([keypoints0[m.queryIdx] for m in good_matches])
        pts1 = np.float32([keypoints1[m.trainIdx] for m in good_matches])

        H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, ransac_threshold)
        inliers_matches = [m for m, flag in zip(good_matches, mask.flatten()) if flag]
    else:
        inliers_matches = good_matches  # 不够点数时不处理

    matched_img = cv2.drawMatches(
        image0, [cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in keypoints0],
        image1, [cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in keypoints1],
        inliers_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    print(f"原始匹配点数: {len(good_matches)}")
    print(f"RANSAC 后保留点数: {len(inliers_matches)}")
    return inliers_matches, matched_img


def superpoint_match(img1_path, img2_path, device='cuda', max_keypoints=-1, distance_ratio=0.85):
    image0 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    image1 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if image0 is None or image1 is None:
        raise FileNotFoundError("图像未找到，请检查路径")
    config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": max_keypoints,
        "remove_borders": 4,
        "path": "weights/superpoint_v1.pth",
        "cuda": (device == 'cuda')    }
    model = SuperPoint(config).to(device)
    model.eval()
    def preprocess(img):
        img = cv2.equalizeHist(img) #直方图均衡化增强对比度
        img_tensor = torch.from_numpy(img / 255.).float()[None, None].to(device)
        return img_tensor

    with torch.no_grad():
        pred0 = model({'image': preprocess(image0)})
        pred1 = model({'image': preprocess(image1)})

    keypoints0 = pred0["keypoints"][0].cpu().numpy()
    keypoints1 = pred1["keypoints"][0].cpu().numpy()
    descriptors0 = pred0["descriptors"][0].cpu().numpy().astype(np.float32).T
    descriptors1 = pred1["descriptors"][0].cpu().numpy().astype(np.float32).T

    bf = cv2.BFMatcher(cv2.NORM_L2)
    print(descriptors0.shape, descriptors1.shape)
    matches = bf.knnMatch(descriptors0, descriptors1, k=2)
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(descriptors0, descriptors1, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < distance_ratio * n.distance:
            good_matches.append(m)
    matched_img = cv2.drawMatches(image0, [cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in keypoints0],
                                  image1, [cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in keypoints1],
                                  good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return good_matches, matched_img


def superglue_match(img1_path, img2_path, device='cuda', max_keypoints=-1, match_threshold=0.2):
    # 读取图像
    image0 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    image1 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if image0 is None or image1 is None:
        raise FileNotFoundError("图像未找到，请检查路径")

    # SuperPoint配置与模型加载
    config_superpoint = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": max_keypoints,
        "remove_borders": 4,
        "path": "weights/superpoint_v1.pth",
        "cuda": (device == 'cuda')}
    superpoint_model = SuperPoint(config_superpoint).to(device)
    superpoint_model.eval()

    def preprocess(img):
        img_tensor = torch.from_numpy(img / 255.).float()[None, None].to(device)
        return img_tensor

    with torch.no_grad():
        pred0 = superpoint_model({'image': preprocess(image0)})
        pred1 = superpoint_model({'image': preprocess(image1)})

    keypoints0 = pred0["keypoints"][0].cpu().numpy()
    keypoints1 = pred1["keypoints"][0].cpu().numpy()
    descriptors0 = pred0["descriptors"][0].cpu().numpy().astype(np.float32)
    descriptors1 = pred1["descriptors"][0].cpu().numpy().astype(np.float32)
    scores0 = pred0["scores"][0].cpu().numpy()
    scores1 = pred1["scores"][0].cpu().numpy()
    # SuperGlue配置与模型加载
    config_superglue = {
        'descriptor_dim': 256,
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': match_threshold,
        'path': 'weights/superglue_outdoor.pth',  # 根据实际情况调整权重文件路径
        'weights': 'outdoor'}
    superglue_model = SuperGlue(config_superglue).to(device)
    superglue_model.eval()
    data = {
        'keypoints0': torch.from_numpy(keypoints0)[None].to(device),
        'keypoints1': torch.from_numpy(keypoints1)[None].to(device),
        'descriptors0': torch.from_numpy(descriptors0)[None].to(device),
        'descriptors1': torch.from_numpy(descriptors1)[None].to(device),
        'scores0': torch.from_numpy(scores0)[None].to(device),
        'scores1': torch.from_numpy(scores1)[None].to(device),
        'image_size0': torch.tensor([[image0.shape[1], image0.shape[0]]]).to(device),
        'image_size1': torch.tensor([[image1.shape[1], image1.shape[0]]]).to(device),}
    with torch.no_grad():
        matches = superglue_model(data)
    good_matches = [(i, matches['matches0'][0, i].item()) for i in range(matches['matches0'].shape[1]) 
                    if matches['matches0'][0, i] > -1]
    matched_img = cv2.drawMatches(
        image0, [cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in keypoints0],
        image1, [cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in keypoints1],
        [cv2.DMatch(_imgIdx=0, _queryIdx=m[0], _trainIdx=m[1], _distance=0) for m in good_matches],
        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matches, matched_img


if __name__ == "__main__":
    img1 = 'datasets_vg/datasets/nwpu/val_0407/queries/000000.jpg'
    img2 =  'datasets_vg/datasets/nwpu/val_0407/references/offset_0_None/000010.tif'

    matches, matched_img = superpoint_match(img1, img2, device='cuda')

    print(f"找到 {len(matches)} 对有效匹配")
    cv2.imwrite("output.jpg", matched_img)