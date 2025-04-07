"""
任务: 生成queries和database尺寸为1000的数据集
      注意该程序需要在tianzhibei环境下运行
时间: 2024/10/22
"""
from osgeo import gdal
import os
from tqdm import tqdm
import random

def Tif_cutting(input_file, output_folder, tile_size, output_format, mode='queries'):
    # 获取文件名和文件夹路径
    file_name = os.path.basename(input_file)
    file_name = file_name.split('.')[0]

    dataset = gdal.Open(input_file)
    if dataset is None:
        return
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    projection = dataset.GetProjection()
    data_type = dataset.GetRasterBand(1).DataType

    # 计算切分数量
    num_tiles_x = width // tile_size
    num_tiles_y = height // tile_size
    if mode == 'database':
        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                # 计算当前切片的左上角坐标
                xoff ,yoff= i * tile_size, j * tile_size 

                data = dataset.ReadAsArray(xoff, yoff, tile_size, tile_size)
                # 如果图像具有多个波段，按波段进行处理
                if bands > 1:    
                    data = [data[b] for b in range(bands)]
                # 构建输出文件名
                output_file = f'{file_name}_{i}_{j}.{output_format}'
                output_file = os.path.join(output_folder, output_file)

                # 创建切片的 GeoTIFF 文件
                driver = gdal.GetDriverByName('GTiff')
                out_dataset = driver.Create(output_file, tile_size, tile_size, bands, data_type)


                # 将数据写入输出文件的每个波段
                if bands > 1:
                    for k in range(bands):
                        out_dataset.GetRasterBand(k + 1).WriteArray(data[k])
                else:
                    out_dataset.GetRasterBand(1).WriteArray(data)
                out_dataset.SetProjection(projection)
                out_dataset.SetGeoTransform((xoff, tile_size, 0, yoff, 0, tile_size))
                # 关闭切片文件
                out_dataset = None
        dataset = None
    elif mode == 'queries':
        # 随机数
        random_number = random.randint(10, 99)
        random.seed(random_number)
        for _ in range(10):
             # 计算当前切片的左上角坐标
            xoff = random.randint(0, width - tile_size)
            yoff = random.randint(0, height - tile_size)

            data = dataset.ReadAsArray(xoff, yoff, tile_size, tile_size)

            # 如果图像具有多个波段，按波段进行处理
            if bands > 1:    
                data = [data[b] for b in range(bands)]

            # 构建输出文件名
            output_file = f'{file_name}_{xoff}_{yoff}.{output_format}'
            output_file = os.path.join(output_folder, output_file)

            # 创建切片的 GeoTIFF 文件
            driver = gdal.GetDriverByName('GTiff')
            out_dataset = driver.Create(output_file, tile_size, tile_size, bands, data_type)
            # 将数据写入输出文件的每个波段
            if bands > 1:
                for k in range(bands):
                    out_dataset.GetRasterBand(k + 1).WriteArray(data[k])
            else:
                out_dataset.GetRasterBand(1).WriteArray(data)
            out_dataset.SetProjection(projection)
            out_dataset.SetGeoTransform((xoff, tile_size, 0, yoff, 0, tile_size))
            # 关闭切片文件
            out_dataset = None
        dataset = None


if __name__ == '__main__':
    Dataset_Dir = '/home3/dataset/tianzhibei/base_map'
    output_dir = '/home3/dataset/tianzhibei/Redal/datasets_vg/datasets/tianzhibei_1000'
    data_class = ['train', 'val', 'test']
    mode = ['database', 'queries']

    # 对数据集进行分割7：2：1=train : test : val
    file_list = [os.path.join(Dataset_Dir, f) for f in os.listdir(Dataset_Dir)]
    train_list, val_list, test_list = file_list[:len(file_list)//10*7], file_list[len(file_list)//10*7:len(file_list)//10*8], file_list[len(file_list)//10*8:]

    for m in mode:
        for cls in data_class:
            output_folder = os.path.join(output_dir, cls, m)
            if os.path.exists(output_folder) == False:
                os.makedirs(output_folder)
            if cls == 'train': 
                for f in tqdm(train_list, desc=f'Processing {cls} data'):
                    Tif_cutting(f, output_folder, 1000, 'png', m)
            elif cls == 'val':
                for f in tqdm(val_list, desc=f'Processing {cls} data'):
                    Tif_cutting(f, output_folder, 1000, 'png', m)
            elif cls == 'test':
                for f in tqdm(test_list, desc=f'Processing {cls} data'):
                    Tif_cutting(f, output_folder, 1000, 'png', m)
        

