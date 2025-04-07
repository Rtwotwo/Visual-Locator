"""
Author: Redal
Date: 2025/04/07
TODO: 制作NWPU0406数据集
Homepage: https://github.com/Rtwotwo/Visual-Locator.git
"""
import os
import cv2
import argparse
from tqdm import tqdm


def nwpu_config():
    parser = argparse.ArgumentParser(description='parameters for nwpu dataset',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument_group('queries folder partition')
    parser.add_argument('--dataset_dir', type=str, default='datasets_vg/datasets/nwpu/val_0406',
                        help='the val_0406 preliminary collated dataset')
    parser.add_argument('--dataset_savedir', type=str, default='datasets_vg/datasets/nwpu/val_0407',
                        help='the name of dataset nwpu should be saved')
    parser.add_argument('--queries_name', type=str, default='queries',
                        help='the folder name of queries')
    parser.add_argument('--queries_savename', type=str, default='queries',
                        help='the folder name of queries nwpu should be saved')
    parser.add_argument('--skip_num', type=int, default=120,
                        help='queries image dataset sparse hop frame number')
    parser.add_argument('--queries_width', type=int, default=512, help='width')
    parser.add_argument('--queries_height', type=int, default=512, help='height')
    parser.add_argument('--queries_csvname', type=str, default='queries.csv',
                        help='the csv file name of queries')
    
    parser.add_argument_group('references folder partition')
    args = parser.parse_args()
    return args
    

class QueryPartition(object):
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
    def forward(self):
        offset_num = len(self.queries_filenames) // self.skip_num * self.skip_num
        for idx, filename in tqdm(enumerate(self.queries_filenames[:offset_num]), desc='partitioning queries'):
            if idx % 120 == 0: 
                frame_path = os.path.join(self.args.dataset_dir, self.args.queries_name, filename)
                frame = cv2.imread(frame_path)
                # 中心裁剪尺寸(512, 512)
                frame = self.__center_clip__(frame)
                cv2.imwrite(os.path.join(self.queries_savepath, f'{idx%120:06d}.jpg'), frame)
    def __center_clip__(self, frame):
        """中心裁剪每一帧的图像数据"""
        h,w = frame.shape[:2]
        crop_w = self.args.queries_width
        crop_h = self.args.queries_height
        start_x, start_y = int(w/2-crop_w/2), int(h/2-crop_h/2)
        cropped_frame = frame[start_y:start_y+crop_h, start_x:start_x+crop_w]
        return cropped_frame


if __name__ == '__main__':
    qp = QueryPartition()
    qp.forward()
        