"""
Author: Redal
Date: 2025/04/07
TODO: 制作NWPU0406数据集
Homepage: https://github.com/Rtwotwo/Visual-Locator.git
"""
import os
import cv2
import math
import argparse
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from tqdm import tqdm
import pandas as pd


def nwpu_config():
    parser = argparse.ArgumentParser(description='parameters for nwpu dataset',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_dir', type=str, default='datasets_vg/datasets/nwpu/val_0406',
                        help='the val_0406 preliminary collated dataset')
    parser.add_argument('--dataset_savedir', type=str, default='datasets_vg/datasets/nwpu/val_0407',
                        help='the name of dataset nwpu should be saved')
    # queries related parameters
    parser.add_argument_group('queries folder partition')
    parser.add_argument('--queries_name', type=str, default='queries',
                        help='the folder name of queries')
    parser.add_argument('--queries_savename', type=str, default='queries',
                        help='the folder name of queries nwpu should be saved')
    parser.add_argument('--skip_num', type=int, default=10,
                        help='queries image dataset sparse hop frame number')
    parser.add_argument('--queries_width', type=int, default=512, help='width')
    parser.add_argument('--queries_height', type=int, default=512, help='height')
    parser.add_argument('--queries_csvname', type=str, default='queries.csv',
                        help='the csv file name of queries')
    # references related parameters
    parser.add_argument_group('references folder partition')
    parser.add_argument('--satellite_dir', type=str, default='datasets_vg/datasets/nwpu/basemap',
                        help='the satellite image path')
    parser.add_argument('--satellite_name', type=str, default='nwpu_small.tif',
                        help='the folder name of references')
    parser.add_argument('--references_savename', type=str, default='references/offset_0_None',
                        help='the folder name of references')
    parser.add_argument('--references_csvname', type=str, default='references.csv',
                        help='references save csv file name')
    # gt_matches related parameters
    parser.add_argument_group('gt_matches folder partition')
    parser.add_argument('--gt_matches_savename', type=str, default='gt_matches.csv',
                        help='the folder name of gt_matches')
    parser.add_argument('--skip_group_num', type=int, default=4,
                        help='the group number of gt_matches')
    args = parser.parse_args()
    return args
    

class QueryPartition(object):
    """首先处理无人机影像数据, 获取queries的图像集以及相应的姿态坐标等位置
    处理好的数据存储位置如下：
    图像数据集: /data2/dataset/Redal/Redal/datasets_vg/datasets/nwpu/val_0407/queries
    姿态数据集: /data2/dataset/Redal/Redal/datasets_vg/datasets/nwpu/val_0407/queries.csv"""
    def __init__(self, args=None, **kwargs):
        self.args = nwpu_config()
        self.skip_num = self.args.skip_num
        # 获取queries图像流的文件名list以及姿态数据的csv
        self.queries_filenames = os.listdir(os.path.join(self.args.dataset_dir, self.args.queries_name))
        self.queries_csvpath = os.path.join(self.args.dataset_dir, self.args.queries_csvname)
        # 获取保存queries图像流的文件夹以及姿态数据的csv
        self.queries_savepath = os.path.join(self.args.dataset_savedir, self.args.queries_savename)
        if not os.path.exists(self.queries_savepath):
            os.makedirs(self.queries_savepath)
            print(f'queries folder has been saved {self.queries_savepath}')
        self.queries_savecsvpath = os.path.join(self.args.dataset_savedir, self.args.queries_csvname)
        # 相关缓存数据
        self.orig_csvdata = pd.read_csv(self.queries_csvpath)
    def forward(self):
        """重新分配queries的图像数据集,按照self.skip_num进行稀疏化
        重新分配queries的csv姿态经纬度数据集"""
        # offset_num = len(self.queries_filenames) // self.skip_num * self.skip_num
        for idx, filename in tqdm(enumerate(self.queries_filenames), desc='partitioning queries'):
            if idx % self.skip_num == 0: 
                frame_path = os.path.join(self.args.dataset_dir, self.args.queries_name, f'{idx:06d}.jpg')
                frame = cv2.imread(frame_path)
                # 中心裁剪尺寸(512, 512)
                frame = self.__center_clip__(frame)
                cv2.imwrite(os.path.join(self.queries_savepath, f'{idx//self.skip_num:06d}.jpg'), frame)  
        # 选择对应的csv_data数据进行处理
        selected_csvdata_list = [idx for idx in range(0, len(self.queries_filenames), self.skip_num)]
        print(selected_csvdata_list)
        selected_csvdata = self.orig_csvdata.iloc[selected_csvdata_list]
        last_column_name = selected_csvdata.columns[-1]
        selected_csvdata[last_column_name] = selected_csvdata[last_column_name].apply(lambda x: f"{int(x.split('.')[0])//self.skip_num:06d}.jpg")
        
        selected_csvdata.to_csv(self.queries_savecsvpath, index=False)
    def __center_clip__(self, frame):
        """中心裁剪每一帧的图像数据"""
        h,w = frame.shape[:2]
        crop_w = self.args.queries_width
        crop_h = self.args.queries_height
        start_x, start_y = int(w/2-crop_w/2), int(h/2-crop_h/2)
        cropped_frame = frame[start_y:start_y+crop_h, start_x:start_x+crop_w]
        return cropped_frame


class ReferencePartition(object):
    def __init__(self, args=None, **kwargs):
        self.args = nwpu_config()
        # 确定各个数据的存储地址路径
        self.satellite_imgpath = os.path.join(self.args.satellite_dir, self.args.satellite_name)
        self.references_savepath = os.path.join(self.args.dataset_savedir, self.args.references_savename)
        if not os.path.exists(self.references_savepath):
            os.makedirs(self.references_savepath)
        self.references_csvpath = os.path.join(self.args.dataset_savedir, self.args.references_csvname)
        self.queries_csvpath = os.path.join(self.args.dataset_savedir, self.args.queries_csvname)
        # 初始化proj的相关transform配置，self.transformer用于utm -> 经纬度
        self.transformer = Transformer.from_crs("epsg:32649", "epsg:4326")
        self.transformer_toutm = Transformer.from_crs("epsg:4326", "epsg:32649")
        self.queries_csvdata = pd.read_csv(self.queries_csvpath)
        # 获取geotiff图像的四个顶点的经纬度
        with rasterio.open(self.satellite_imgpath) as src:
            bounds = src.bounds
            src_crs = src.crs
            transformer_crs = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
            left, bottom, right, top = bounds
            corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
            # 左上角、右上角、右下角、左下角
            self.corners = [transformer_crs.transform(*corner) for corner in corners]
            self.length_longitude = self.corners[1][0] - self.corners[0][0]
            self.length_latitude = self.corners[0][1] - self.corners[2][1]
    def forward(self):
        with rasterio.open(self.satellite_imgpath) as src:
            # make utm_center coordinate formatclass: (easting, northing, name)
            utm_center = self.queries_csvdata.iloc[:, :]
            utm_center = utm_center.apply(lambda row: (row['easting'], row['northing'], row['name']), axis=1).tolist()
            # 选择每四张作为一组进行分析,单独提出每组第一张作为references
            utm_center_first = utm_center[0::self.args.skip_group_num]
            for utm in tqdm(utm_center_first, desc='partitioning references'):
                window = self.__get_window__(src, (utm[0], utm[1]))
                img_array = src.read(window=window)
                transform = src.window_transform(window)
                # 保存分割的文件geo_tiff格式
                meta = src.meta.copy()
                meta.update({
                    "height": self.args.queries_height,
                    "width": self.args.queries_width,
                    "transform": transform})
                # 创建保存文件路径
                geotiff_savepath = os.path.join(self.references_savepath, f"{int(utm[2].split('.')[0])//self.args.skip_group_num:06d}.tif")
                with rasterio.open(geotiff_savepath, 'w', **meta) as dst:
                    dst.write(img_array)  
        # 保存references的图像对应数据坐标references.csv文件
        references_csvdata = {'easting': self.queries_csvdata['easting'].iloc[::self.args.skip_group_num], 
                              'northing': self.queries_csvdata['northing'].iloc[::self.args.skip_group_num],
                              'name': self.queries_csvdata['name'].iloc[::self.args.skip_group_num].apply(lambda x: 
                                'offset_0_None/'+f"{int(x.split('.')[0])//self.args.skip_group_num:06d}.tif")}
        references_csvdata = pd.DataFrame(references_csvdata)
        references_csvdata.to_csv(self.references_csvpath, index=False)
    def __get_window__(self, src, utm_center):
        """获取需要裁剪的卫星影像的裁剪窗口
        :param src: 输入的satallite数据源
        :param utm_center: 输入的utm坐标中心点"""
        orig_width, orig_height = src.width, src.height
        # 将utm转为经纬度进行线性插值确定像素坐标
        latitude , longitude = self.transformer.transform(utm_center[0], utm_center[1])
        center_x = int((longitude - self.corners[0][0]) / self.length_longitude * orig_width)
        center_y = int((self.corners[0][1] - latitude) / self.length_latitude * orig_height)
        # 计算裁剪左上角起点
        col_start = center_x - self.args.queries_width // 2
        row_start = center_y - self.args.queries_height // 2
        window = Window(col_start, row_start, self.args.queries_width*1.5, self.args.queries_height*1.5)
        return window
    def __gt_matches__(self):
        """计算utm坐标与经纬度坐标的匹配关系,生成GT_Matches.csv文件"""
        queries_csvdata = pd.read_csv(self.queries_csvpath)
        references_csvdata = pd.read_csv(self.references_csvpath)
        self.gt_matches_data = {}
        query_ind, query_name, ref_ind, ref_name, distance = [], [], [], [], []
        # 直接使用csv文件的数据进行操作
        for q_id in range(len(queries_csvdata)):
            query_ind.append(q_id)
            query_name.append(queries_csvdata['name'].iloc[q_id])
            ref_ind.append( q_id // self.args.skip_group_num)
            ref_name.append(references_csvdata['name'].iloc[q_id // self.args.skip_group_num])
            distance.append(math.sqrt((queries_csvdata['easting'].iloc[q_id] - references_csvdata['easting'].iloc[q_id // self.args.skip_group_num])**2 +
                                                (queries_csvdata['northing'].iloc[q_id] - references_csvdata['northing'].iloc[q_id // self.args.skip_group_num])**2) )
        # 保存gt_matches.csv文件
        self.gt_matches_data = {'query_ind': query_ind, 'query_name': query_name, 'ref_ind': ref_ind, 'ref_name': ref_name, 'distance': distance}
        gt_matches_csvpath = os.path.join(self.args.dataset_savedir, self.args.gt_matches_savename)
        gt_matches_csvdata = pd.DataFrame(self.gt_matches_data)
        gt_matches_csvdata.to_csv(gt_matches_csvpath, index=False)
        
        
if __name__ == '__main__':
    qp = QueryPartition()
    qp.forward()
    rp = ReferencePartition()
    rp.forward()
    rp.__gt_matches__()