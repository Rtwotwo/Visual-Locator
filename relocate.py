"""
Author: Redal
Date: 2025-12-18
Todo: relocate for uav images to compute utm data
Command: nohup python relocate.py > all.log 2>&1
Homepage: https://github.com/Rtwotwo/Visual-Locator.git
"""
import os
import torch
import cv2
import json
import csv
import math
import numpy as np
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt

import rasterio
import argparse
from tqdm import tqdm
from rasterio.transform import xy
from pyproj import Transformer
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


def draw_matches_custom(img1, img2, mkpts0, mkpts1, color='lime', line_width=1):
    """使用matplotlib绘制LoTFR的匹配结果
    mkpts0/mkpts1:匹配点坐标列表变量"""
    if img1.ndim == 3:img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    else:img1 = img1  
    if img2.ndim == 3:img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    else:img2 = img2
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    w = w1 + w2
    # 创建拼接图像
    if img1.ndim == 2:
        out_img = np.zeros((h, w), dtype=np.uint8)
        out_img[:h1, :w1] = (img1 * 255).astype(np.uint8)
        out_img[:h2, w1:] = (img2 * 255).astype(np.uint8)
        out_img = np.stack([out_img] * 3, axis=-1)  # 转为 RGB
    else:
        out_img = np.zeros((h, w, 3), dtype=np.uint8)
        out_img[:h1, :w1] = (img1 * 255).astype(np.uint8)
        out_img[:h2, w1:] = (img2 * 255).astype(np.uint8)
    # 绘制匹配线并可视化曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(out_img)
    ax.axis('off')
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        ax.plot([x0, x1 + w1], [y0, y1], color=color, linewidth=line_width, alpha=0.8)
        plt.tight_layout()
    return fig, ax


def get_center_utm(tif_path):
    """生成GeoTIFF图像中心点的UTM坐标
    utm_x/utm_y:中心点的UTM坐标值也即是easting和northing"""
    with rasterio.open(tif_path) as src:
        height, width = src.height, src.width
        center_row = height / 2.0
        center_col = width / 2.0
        center_x, center_y = xy(src.transform, center_row, center_col)
        crs = src.crs.to_string() if src.crs else None
        # 如果源数据不是EPSG:4326，则转换之
        transformer_to_wgs84 = None
        if crs != "EPSG:4326":
            transformer_to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            center_x, center_y = transformer_to_wgs84.transform(center_x, center_y)
        # 将经纬度转换为指定区域的UTM坐标
        transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
        utm_x, utm_y = transformer_to_utm.transform(center_x, center_y)
        return utm_x, utm_y


def get_any_pixels_utm(tif_path, y_pixel, x_pixel):
    """获取卫星影像中任意位置的UTM的坐标"""
    with rasterio.open(tif_path) as src:
        # 将像素坐标(col, row)转为地图坐标
        map_x, map_y = src.xy(y_pixel, x_pixel)  
        crs = src.crs
    if crs is None: raise ValueError("[WARNING] GeoTIFF缺少CRS信息!")
    transformer_to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer_to_wgs84.transform(map_x, map_y)
    # 将经纬度转换为西安所在的 UTM 坐标（UTM Zone 49N → EPSG:32649）
    transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    return transformer_to_utm.transform(lon, lat)


def compute_utm_for_basemap(basemap_paths):
    """计算出所有的basemap图像的UTM坐标位置,并保存在字典中返回
    basemap_paths:所有basemap图像的路径列表
    同时也返回basemap_utm_dict的相关数据"""
    basemap_utm_dict = {}
    for basemap_path in tqdm(basemap_paths, desc=f"[INFO] 计算Basemap图像UTM坐标:"):
        utm_x, utm_y = get_center_utm(basemap_path)
        basemap_utm_dict[basemap_path] = [utm_x, utm_y]
    with open("basemap_utm.json", 'w') as bf:
        json.dump(basemap_utm_dict, bf, indent=4)
    return basemap_utm_dict


def read_utm_for_queries(file_path):
    """从给定路径读取CSV文件,并返回包含所有
    basemap图像及其相关信息的列表"""
    basemap_info_list = []
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)
        # 将读取的数据转换为合适的类型
        for row in reader:
            data = {'easting': float(row[0]),
                    'northing': float(row[1]),
                    'altitude': float(row[2]),
                    'orient_x': float(row[3]),
                    'orient_y': float(row[4]),
                    'orient_z': float(row[5]),
                    'orient_w': float(row[6]),
                    'name': row[7]}
            basemap_info_list.append(data)
    return basemap_info_list


def find_nearest_basemap(query_utm, basemap_utms):
    """送入一对utm坐标,从basemap_utms中找到最近的basemap切片
    Return 返回最近的basemap的图片文件的路径
    注意utm坐标的组织形式[easting, northing]"""
    nearest_fp = ''
    nearest_dis = float('inf')  # 更安全的写法
    for fp in basemap_utms.keys():
        current_utm = basemap_utms[fp]
        current_dis = math.sqrt((current_utm[0] - query_utm[0])**2 + 
                                (current_utm[1] - query_utm[1])**2)
        if current_dis < nearest_dis:
            nearest_dis = current_dis
            nearest_fp = fp
    if nearest_fp == '': raise ValueError('[WARNING] 注意没有找到最佳匹配的Basemap的切片!')
    else: return nearest_fp


def lofter_match(img1_path, img2_path, loftr):
    """使用lofter进行两张图像的特征匹配,返回匹配点对数量
    img1_path/img2_path图像保存的路径位置
    return: 经过LoFTR匹配后的点对数量的整数值,以及匹配点坐标mkpts0,mkpts1"""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    assert img1 is not None and img2 is not None, \
        f"[WARNING] 图像读取失败,请检查路径: {img1_path}和{img2_path}!"
    img1 = K.image_to_tensor(img1, keepdim=False).float() / 255.
    img2 = K.image_to_tensor(img2, keepdim=False).float() / 255.
    gray1 = K.color.rgb_to_grayscale(img1)
    gray2 = K.color.rgb_to_grayscale(img2)
    # 设置为评估模式进行匹配
    with torch.no_grad():
        input_dict = {"image0": gray1, "image1": gray2}
        correspondences = loftr(input_dict)
    # 统计匹配的点对数量
    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    return len(mkpts0), mkpts0, mkpts1


def compute_homography(mkpts0, mkpts1, ransac_thresh=3.0):
    """使用匹配点计算从img0无人机到img1卫星图的单应性矩阵 H
    mkpts0: [N, 2] 无人机图像关键点 (x, y)
    mkpts1: [N, 2] 卫星图像关键点 (x, y)
    ransac_thresh: RANSAC 内点阈值（像素）
    H: 3x3 单应性矩阵(np.ndarray),若失败返回None
    inlier_mask: 内点掩码"""
    if len(mkpts0) < 4:
        print(f"[WARNING] 匹配点不足4个,当前仅{len(mkpts0)}个无法计算H矩阵!")
        return None, None
    pts0 = mkpts0.astype(np.float32)
    pts1 = mkpts1.astype(np.float32)
    # 使用RANSAC计算 H
    H, mask = cv2.findHomography(
        srcPoints=pts0,
        dstPoints=pts1,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh)
    inliers = mask.ravel().astype(bool) if mask is not None else None
    return H, inliers


def project_uavcenter_with_H(H, uav_img_shape):
    """将无人机图像中心点通过H投影到卫星图坐标系
    返回无人机图像中心点经过单应性变换后,在卫星底图图像坐标系中的像素位置"""
    h, w = uav_img_shape
    center = np.array([[w / 2, h / 2]], dtype=np.float32)  
    center_homo = np.hstack([center, np.ones((1, 1))])  
    proj_homo = (H @ center_homo.T).T  
    proj = proj_homo[:, :2] / proj_homo[:, 2:3]
    return proj[0] 


def main():
    """LoFTR特征匹配器类: 初始化LoFTR模型并导入预训练权重,提供匹配接口并实现
    为每个queries文件夹的图像计算绝对的UTM坐标位置并保存在csv文件中"""
    # 配置函数参数解析器
    parser = argparse.ArgumentParser(description="LoFTR Matcher for Aerial Images")
    parser.add_argument('--queries_path', type=str, default="./queries/", help="Path to queries")
    parser.add_argument('--queries_csv_path', type=str, default="./queries.csv", help="Path to queries csv file for fundamental data")
    parser.add_argument('--basemap_path', type=str, default="./basemap/", help="Path to basemap")
    parser.add_argument('--ckpt_path', type=str, default="./weights/outdoor_ds.ckpt", help="Path to LoFTR checkpoint")
    args = parser.parse_args()

    # 初始化LoFTR匹配器并导入权重文件outdoor_ds.ckpt
    loftr = KF.LoFTR(pretrained=None)
    checkpoint = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
    if "state_dict" in checkpoint:state_dict = checkpoint["state_dict"]
    else:state_dict = checkpoint
    loftr.load_state_dict(state_dict, strict=True)
    loftr.eval()
    # 获取queries和basemap的图像存储路径,注意queries_paths的排序比较重要
    queries_paths = [os.path.join(ROOT_PATH, 'queries', fn) for fn in os.listdir(args.queries_path)]
    basemap_paths = [os.path.join(ROOT_PATH, 'basemap', fn) for fn in os.listdir(args.basemap_path)]
    # 计算所有basemap图像的UTM坐标位置并保存
    queries_utms = read_utm_for_queries(args.queries_csv_path)
    basemap_utms = compute_utm_for_basemap(basemap_paths)
    # 处理无人机中心位置的utm数据
    results = []
    for idx in range(len(queries_paths)):
        query_path = os.path.join(ROOT_PATH, 'queries', f'{idx:06d}.jpg')
        print(f'[INFO] 正在处理queries文件夹中的图片: {query_path}!')
        # 1.计算距离query_path最近的basemap的切片
        query_utm = [queries_utms[idx]['easting'], queries_utms[idx]['northing']]
        nearest_basemap_fp = find_nearest_basemap(query_utm, basemap_utms)
        # 2.计算映射的H矩阵参数
        _, mkpts0, mkpts1= lofter_match(query_path, nearest_basemap_fp, loftr)
        H, _ = compute_homography(mkpts0, mkpts1)
        query_img = cv2.imread(query_path)
        h, w = query_img.shape[:2]
        center_proj = project_uavcenter_with_H(H, [h, w])
        # 3.获取真实的utm的坐标
        x_sat, y_sat = center_proj
        utm_x, utm_y = get_any_pixels_utm(nearest_basemap_fp, y_sat, x_sat)
        print(f'[INFO] 最匹配的basemap的切片: {nearest_basemap_fp}, 映射后的像素位置xy: {center_proj}!')
        print(f'[INFO] 当前的{query_path}文件中心的UTM坐标: (easting:{utm_x},northing:{utm_y})!\n\n')
        basename = os.path.basename(query_path)
        results.append((utm_x, utm_y, basename))

    output_csv = "relocated_queries.csv"
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['easting', 'northing', 'name'])
        for easting, northing, name in results:
            writer.writerow([f"{easting:.6f}", f"{northing:.6f}", name])
    print(f"[INFO] 所有结果已保存至{output_csv}!")

    # # 读取无人机航拍图像和大比例尺底图进行匹配
    # img1_path = os.path.join(ROOT_PATH, "queries/000000.jpg")
    # img2_path = os.path.join(ROOT_PATH, "basemap/000200.tif")
    # _, xy1, xy2 = lofter_match(img1_path, img2_path, loftr)
    # fig, ax = draw_matches_custom(cv2.imread(img1_path), cv2.imread(img2_path), xy1, xy2)
    # plt.savefig("matches_1.jpg", dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    main()