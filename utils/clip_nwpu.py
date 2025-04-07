"""
任务: 完成对nwpu校园卫星影像的分割处理
      分别包含small/big/large三种影像数据,
      注意采用的geotiff格式使用Web墨卡托投影
时间: 2025/03/02-Redal
"""
import os
import cv2
import numpy as np
import argparse
import rasterio
import datetime
import pandas as pd
from PIL import Image
from rasterio.windows import Window
from pyproj import Transformer
from pyproj import Proj, transform



######################  定义变量解析阈  ######################
def config():
    parser = argparse.ArgumentParser(description='define the related clipping arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--clip_width', type=int, default=500,
                        help='clipping the geotiff with clip_width')
    parser.add_argument('--clip_height', type=int, default=500,
                        help='clipping the geotiff with clip_height')
    parser.add_argument('--geo_type', type=str, default='small',
                        help='there are three types small/big/large')
    parser.add_argument('--dataset_name', type=str, default='nwpu',
                        help='choose the nwpu dataset to clip and test')
    parser.add_argument('--dataset_dir', type=str, default='/data2/dataset/Redal/Redal/datasets_vg/datasets',
                        help='the nwpu dataset located at the above direction')
    parser.add_argument('--save_mode', type=str, default='database',
                        help='the satellite imagery can be ')
    parser.add_argument('--clip_side', type=str, default=300, 
                        help='because of the black side of the image, so clip the side image')
    parser.add_argument('--is_clip', type=bool, default=True,
                        help='whether clip the geotiff images black side')
    parser.add_argument('--save_cls', type=str, default='train',
                        help='you can choose train/test/val')
    
    # Processing drone data
    parser.add_argument('--uav_video', type=str, default='video', help="the uav video's directory")
    parser.add_argument('--video_name', type=str, default='uav_video.mp4', help='the video of uav')
    parser.add_argument('--attitude', type=str, default='uav_attitude', help='the attitude data name, plus .txt/.csv')
    parser.add_argument('--video_save', type=str, default='queries', help='convert video to images and save folder')
    parser.add_argument('--video_folder', type=str, default='val', help='the dataset save mode folder val')
        
    args = parser.parse_args()
    return args



######################  完成nwpu卫星地图切割函数划分  ######################
def clip_geotiff(args):
    """clip the geotiff file with the related requirements"""
    input_filepath = os.path.join(args.dataset_dir, args.dataset_name, 'basemap',
                    'nwpu_'+args.geo_type+'.tif')
    img_width, img_height, _ = cv2.imread(input_filepath).shape
    w_num = int(img_width / args.clip_width)
    h_num = int(img_height / args.clip_height)
    print(f'the geotiff is {img_width}x{img_height}')
    print(f'the geotiff has{w_num}x{h_num} just are {w_num*h_num} patches')
    if args.is_clip:
        # if clip the black side, you can set is_clip is true
        clip_side(args)    
    for i in range(w_num):
        for j in range(h_num):
            with rasterio.open(input_filepath) as src:
                window = Window(col_off=i*500, row_off=j*500,
                                width=args.clip_width, height=args.clip_height)
                subset = src.read(window=window) # read the window data
                meta = src.meta.copy()
                meta.update({
                    'height':window.height,
                    'width':window.width,
                    'transform':rasterio.windows.transform(window, src.transform)})
                # create the save file and dataset working as database
                output_dir = os.path.join(args.dataset_dir, args.dataset_name,
                                    args.save_cls, args.save_mode)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                output_filepath = os.path.join(output_dir, f"{args.geo_type}_{i}_{j}.tif")
                with rasterio.open(output_filepath, 'w', **meta) as dst:
                    dst.write(subset) # show current time 
                    current_time = datetime.datetime.now()
                    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f'{formatted_time}:\t{output_filepath} has been saved...', end='\r', flush=True)
                    

def process_latitude_longitude(filepath, only_center=False):
    """decoding each tif format image's coordinate information
    :param filepath: the remote sensing image tiff format coordinate information
    :param center: Whether to get only the coordinate information of the center of the image"""
    with rasterio.open(filepath) as src:
        transform = src.transform
        width, height = src.width, src.height
        center_x, center_y = int(width / 2), int(height / 2)
        # Web Mercator projection converted to latitude and longitude
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326")
        if only_center:
            lon, lat = transform * (center_x, center_y)
            lon_wgs84, lat_wgs84 = transformer.transform(lon, lat)
            return lon_wgs84, lat_wgs84
        elif not only_center:
            raise RuntimeError(f'Global conversion is not supported!')
        

def clip_side(args):
    """clip the side image and set length"""
    input_filepath = os.path.join(args.dataset_dir, args.dataset_name, 'basemap',
                        f"nwpu_{args.geo_type}.tif")
    with rasterio.open(input_filepath) as src:
        image = src.read()
        transform = src.transform
        profile = src.profile
    if len(image.shape) == 3 and image.shape[0] == 3: 
        image = image.transpose((1, 2, 0))
    else: pass
    # check the non black area
    non_black_mask = np.any(image != [0, 0, 0], axis=-1)
    rows, cols = np.where(non_black_mask)
    top_left_row = np.min(rows)
    top_left_col = np.min(cols)
    bottom_right_row = np.max(rows)
    bottom_right_col = np.max(cols)
    cropped_image = image[top_left_row:bottom_right_row+1, 
                    top_left_col:bottom_right_col+1]
    # open the geo_tiff 
    transform = rasterio.Affine(
        transform.a,
        transform.b,
        transform.c + top_left_col * transform.a,
        transform.d,
        transform.e,
        transform.f + top_left_row * transform.e)
    # Update the configuration file to save the cropped image
    profile.update(transform=transform, width=bottom_right_col - top_left_col + 1,
                    height=bottom_right_row - top_left_row + 1)
    output_filepath = os.path.join(args.dataset_dir, args.dataset_name, 'basemap',
                    f"nwpu_{args.geo_type}.tif")
    with rasterio.open(output_filepath, 'w', **profile) as dst:
            dst.write(cropped_image.transpose((2,0,1)))
            

def remove_allblack(args):
    """remove all the all black images"""
    input_dir = os.path.join(args.dataset_dir, args.dataset_name, 
                args.save_cls, args.save_mode)
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if file_path.lower().endswith(('.tif', '.jpg', '.png')):
            with Image.open(file_path) as img:
                img_array = np.array( img.convert('L') ) 
                if np.all(img_array == 0): 
                    print(f'the image path:{file_path} is all black') 
                    os.remove(file_path) 
                else: pass          
                                     


######################  完成nwpu无人机视频处理函数  ######################
def txt_to_excle(txt_path, output_path):
      """read drone attitude data"""
      df = pd.read_csv(txt_path)
      df.to_csv(output_path, index=False)
      
def get_video_frame_count(video_path):
    """Get the total frame number of the video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


class VideoToData:
    """Convert video data into images and save the image data of 
    each frame of video separately in a csv file"""
    def __init__(self, args, **kwargs):
        self.args = args
        self.video_path = os.path.join(args.dataset_dir, args.dataset_name,
                        args.uav_video, args.video_name)
        self.videotofolder = os.path.join(args.dataset_dir, args.dataset_name,
                        args.video_folder, args.video_save)
        if not os.path.exists(self.videotofolder):
            os.mkdir(self.videotofolder)
        self.video_cap = cv2.VideoCapture(self.video_path)
        # altittude data file
        self.csv_filepath = os.path.join(args.dataset_dir, args.dataset_name,
                        args.uav_video, f'{args.attitude}.csv')
        self.csv_savepath = os.path.join(args.dataset_dir, args.dataset_name,
                        args.video_folder, f'queries.csv')
        # database data
        self.database_dir = os.path.join(args.dataset_dir, args.dataset_name, args.video_folder, 'database')
        self.database_savedir = os.path.join(args.dataset_dir, args.dataset_name, args.video_folder, 'references')
        if not os.path.exists(self.database_savedir):
            os.mkdir(self.database_savedir)
        self.database_savepath = os.path.join(args.dataset_dir, args.dataset_name,
                        args.video_folder, f'references.csv')
    def __to_img__(self):
        frame_count = 0
        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret :
                frame_filepath = os.path.join(self.videotofolder, f'{frame_count:06d}.jpg')
                cv2.imwrite(frame_filepath, frame)
                print(f'{frame_filepath} has been saved')
                frame_count += 1
            else: break
    def __save_queries_altittude__(self):
        wgs84 = Proj(proj='latlong', datum='WGS84')
        # utm = Proj(proj='utm', zone=49, datum='WGS84')
        center_lon = 108.9  # 西安中心经度，可改为从metadata读取
        utm_zone = self.get_utm_zone(center_lon)
        utm = Proj(proj='utm', zone=utm_zone, datum='WGS84')
        altittude_data = {}
        data = pd.read_csv(self.csv_filepath)
        longitude, latitude = data['gps_longitude'] / 10**7, data['gps_latitude'] / 10**7
        easting, northing = transform(wgs84, utm, longitude, latitude)
        
        altittude_data['easting'] = easting
        altittude_data['northing'] = northing
        altittude_data['altitude'] = data['gps_altitude'] / 10**3
        altittude_data['orient_x'] = data['Quaternion x'] 
        altittude_data['orient_y'] = data['Quaternion y']
        altittude_data['orient_z'] = data['Quaternion z']
        altittude_data['orient_w'] = data['Quaternion w']
        altittude_data['name'] = [f'{idx:06d}.jpg' for idx in range(24088)] # 8994
        
        df = pd.DataFrame(altittude_data)
        df.to_csv(self.csv_savepath, index=False)
        print(f'{self.csv_savepath} has been saved')

    def __database_csvfile__(self):
        data = {}
        easting, northing, name = [], [], []
        count_num = 0
        wgs84 = Proj(proj='latlong', datum='WGS84')
        # utm = Proj(proj='utm', zone=49, datum='WGS84')
        # 动态计算分带（假设卫星图中心点经度已知）
        center_lon = 108.9  # 西安中心经度，可改为从metadata读取
        utm_zone = self.get_utm_zone(center_lon)
        utm = Proj(proj='utm', zone=utm_zone, datum='WGS84')
        for i in range(20):
            for j in range(19):
                geotiff_filepath = os.path.join(self.database_dir, f'small_{i}_{j}.tif')
                geotiff_savepath = os.path.join(self.database_savedir, f'{count_num:06d}.jpg')
                with rasterio.open(geotiff_filepath) as dataset:
                    # Get the boundaries of the image
                    bounds = dataset.bounds
                    # Calculate the center point coordinates (Easting, Northing)
                    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326")
                    center_easting, center_northing = (bounds.left + bounds.right) / 2, (bounds.bottom + bounds.top) / 2
                    center_lat, center_lon = transformer.transform(center_easting, center_northing)
                    img_easting, img_northing = transform(wgs84, utm, center_lon, center_lat)
                    
                    easting.append(img_easting)
                    northing.append(img_northing)
                    name.append(f'{count_num:06d}.jpg')
                cv2.imwrite(geotiff_savepath, cv2.imread(geotiff_filepath))
                print(f'{geotiff_savepath} has been saved reference directory')
                count_num += 1
        data['easting'] = easting
        data['northing'] = northing
        data['name'] = name
        df = pd.DataFrame(data)
        df.to_csv(self.database_savepath, index=False)
        print(f'{self.database_savepath} has been saved')
    def get_utm_zone(self, longitude):
        """动态计算UTM分带编号"""
        return int(longitude // 6 + 31)  # 北半球计算公式
        
    
        
        
                        
######################  主函数测试分析  ################################
#  python clip_nwpu.py --is_clip=True --save_mode=database             #
# --save_cls=train --geo_type=large --clip_width=512 --clip_height=512 #
########################################################################
if __name__ == '__main__':
    args = config()
    # clip satellite map
    # clip_geotiff(args)
    # remove_allblack(args)
    
    # split the uav video
    vtd = VideoToData(args)
    # vtd.__to_img__()
    vtd.__save_queries_altittude__()
    vtd.__database_csvfile__()
    