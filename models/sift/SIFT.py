"""
任务: 使用SIFT算法提取图像特征点以及描述符
      进行遥感+无人机连续帧视角配准
时间: 2025/01/23-Redal
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def sift_match(input_img1_path, input_img2_path, dis_threshold=0.75):
    """使用SIFT算法提取图像特征点以及描述符 - 单张图像配准
    进行遥感+无人机连续帧视角配准
    :param input_img1: 输入配准图像,要求为gray格式
    :param input_img2: 输入配准图像,要求为gray格式"""
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    input_img1 = cv2.imread(input_img1_path, cv2.IMREAD_GRAYSCALE)
    input_img2 = cv2.imread(input_img2_path, cv2.IMREAD_GRAYSCALE)
    keypoints1, descriptors1 = sift.detectAndCompute(input_img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(input_img2, None)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    # 应用比率测试以过滤匹配
    good_matches = []
    for m, n in matches:
        if m.distance < dis_threshold * n.distance:
            good_matches.append(m)
    # 绘制匹配结果并返回
    matched_img = cv2.drawMatches(input_img1, keypoints1, input_img2, keypoints2, 
                good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return good_matches, matched_img


def sift_match_pro(input_img1_path, input_img2_path, dis_threshold=0.75):
    """使用SIFT算法提取图像特征点以及描述符 - 单张图像配准
    进行遥感+无人机连续帧视角配准
    :param input_img1_path: 输入配准图像路径,要求为gray格式
    :param input_img2_path: 输入配准图像路径,要求为gray格式"""
    # sift = cv2.SIFT_create(nfeatures=0,  # 特征点的最大数量
    #                        nOctaveLayers=3,  # 每个倍频程的层数
    #                        contrastThreshold=0.01,  # 对比度阈值
    #                        edgeThreshold=25,  # 边缘阈值
    #                        sigma=1.4)  # 高斯滤波器标准差
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    # 加载图像并转换为灰度图像
    input_img1 = cv2.imread(input_img1_path, cv2.IMREAD_GRAYSCALE)
    input_img2 = cv2.imread(input_img2_path, cv2.IMREAD_GRAYSCALE)
    keypoints1, descriptors1 = sift.detectAndCompute(input_img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(input_img2, None)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < dis_threshold * n.distance:
            good_matches.append(m)
    if len(good_matches) > 4:  # 至少需要4个点来计算单应性矩阵
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        
        # 增大 reproject error 阈值（如从5.0→35.0）
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC , 40.0)
        matches_mask = mask.ravel().tolist()
        good_matches = [m for m, mk in zip(good_matches, matches_mask) if mk == 1]
        matched_img = cv2.drawMatches(input_img1, keypoints1, input_img2, keypoints2,
                                      good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), 4))
        matches_mask = None
        matched_img = cv2.hconcat([input_img1, input_img2])
    return good_matches, matched_img


def sift_candidate_match(query_img_path, candi_img_paths, dis_threshold=0.7):
    """使用SIFT算法提取图像特征点以及描述符 - 多张候选图像配准
    进行遥感+无人机连续帧视角配准
    :param query_img_path: 输入配准图像路径,要求为gray格式
    :param candi_img_paths: 输入配准图像路径列表,要求为gray格式
    :param dis_threshold: 匹配点计算阈值
    :param best_match_path: 返回最佳匹配的候选图像保存路径"""
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    query_grayimg = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)
    query_kp , query_dp = sift.detectAndCompute(query_grayimg, None)
    # 提取查询图像的描述符以及特征
    candi_kp, candi_dp = [], []
    for candi_path in candi_img_paths:
        candi_grayimg = cv2.imread(candi_path, cv2.IMREAD_GRAYSCALE)
        kp , dp = sift.detectAndCompute(candi_grayimg, None)
        candi_kp.append(kp); candi_dp.append(dp)
    # 初始化FLANN或BFMatcher匹配器
    FLANN_INDEX_KDTREE = 1
    index_params= dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = []
    for desc in candi_dp: # 遍历所有候选图像的描述符
        matches.append(flann.knnMatch(query_dp, desc, k=2))
    # 相似性匹配得分投票
    similarity_score = []
    for match in matches:
        good_matches = []
        for m,n in match:
            if m.distance < dis_threshold * n.distance:
                good_matches.append(m)
        similarity_score.append(len(good_matches))
    best_match_index = similarity_score.index(max(similarity_score))
    best_match_path = candi_img_paths[best_match_index]
    return best_match_path
        

def sift_match_show(matched_img):
    """使用matplotlib显示配准结果
    :param matched_img: 输入配准图像,要求为gray格式"""
    plt.imshow(matched_img)
    plt.show()
    