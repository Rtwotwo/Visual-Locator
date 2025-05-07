"""
任务: 处理自定义nwpu数据集,将queries和reinference文件夹数据
      映射在数据空间,封装SelaVPR映射的embedding索引和地址
时间: 2025/03/03-Redal
"""
import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(root_dir)

import torch
from functools import partial
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import transforms
from models.selavpr import network
from models.prithvi.Prithvi_ViT import RetrievalViT

selavpr_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


######################### 定义变量解析 ###########################
def parsers():
    parser = argparse.ArgumentParser(description = 'Describe the parameters of the running program', 
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_dir', type=str, default='/data2/dataset/Redal/Redal/datasets_vg/datasets',
                        help='please write down the correct dataset direction')
    parser.add_argument('--dataset_name', type=str, default='tianzhibei', 
                        help="there are two datasets you can choose ['tianzhibei', 'UAV']")
    parser.add_argument('--dataset_proc', type=str, default='train',
                        help="there are some choices ['train', 'test', 'val']")
    parser.add_argument('--dataset_mode', type=str, default='references',
                        help='test:database train:database/queries val:database/queries')
    parser.add_argument('--dist_offset', type=str, default='offset_0_None',
                        help='you can choose offset_{0/20/40}_{North/South}')
    parser.add_argument('--model_dir ', type=str, default='/data2/dataset/Redal/Redal/models',
                        help="please write down the correct model direction")
    parser.add_argument('--model_name', type=str, default='selavpr', 
                        help="there are two retrieval model names you can choose ['selavpr', 'prithvi_100m']")
    parser.add_argument('--weight_dir', type=str, default='/data2/dataset/Redal/Redal/weights',
                        help='please write down the correct weight direction')
    parser.add_argument('--weight_name', type=str, default='dinov2_vitl14_pretrain.pth',
                        help="there are some choices ['dinov2_vitl14_pretrain.pth', 'Prithvi_100M.pt']")
    parser.add_argument('--embed_dim', type=int, default=1024, 
                        help='the embed_dim should satisfies the model input need')
    parser.add_argument('--save_dir', type=str, default='/data2/dataset/Redal/Redal/embedding',
                        help='save the remote sensing image embedding direction')
    parser.add_argument('--save_name', type=str, default=None,
                        help='you should make sure the correct save file name')
    parser.add_argument('--save_mode', type=str, default='database',
                        help='test:database train:database/queries val:database/queries')
    args = parser.parse_args()
    return args
    
    
    
########################### 定义数据处理器 #######################
class AltoDataset(Dataset):
    """ALTO数据集映射到数据空间以便检索任务"""
    def __init__(self, args, transform=None, **kwargs,):
        self.args = args
        self.transform = transform
        self.embed_dim = args.embed_dim
        self.weight_path = os.path.join(args.weight_dir, args.weight_name)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataset_model_path = os.path.join(args.dataset_dir, 
                            args.dataset_name, args.dataset_proc)
        
        if args.model_name == 'selavpr':
            self.vit_model = network.GeoLocalizationNet(weight_path=self.weight_path).to(self.device)
        elif args.model_name == 'prithvi_100m':
            # Prithvi_100M model input image should satisfied with 6 channels, depend on the self.chan_proj
            self.chan_proj = nn.Conv2d(in_channels=3, out_channels=6)
            self.vit_model = RetrievalViT(img_size=224, patch_size=16,
                    num_frames=1, tubelet_size=1,
                    in_chans=6, embed_dim=768, depth=12, num_heads=12,
                    mlp_ratio=4., norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                    norm_pix_loss=False, weights=self.weight_path).to(self.device)
        else:raise  ValueError('There is satisfied model class to build vit_model')
    def __len__(self):
        return len(self.dataset_model_path)
    def __getitem__(self):
        # specified the code for nwpu dataset
        self.embeddings, self.address = [],[]
        self.alto_embed_paths = None      
        if self.args.dataset_proc == 'val_0407':
        # if self.args.dataset_proc == 'train':
            # the dataset_model_path(parent dir) with children dirs
            database_query_dirs = os.listdir(self.dataset_model_path)
            for dir in database_query_dirs:
                # 如果使用nwpu dataset, 需要将database换成references/offset_0_None
                if dir == 'references' and dir == self.args.dataset_mode:
                    full_dir = os.path.join(self.dataset_model_path,'references/offset_0_None')
                    for file in tqdm(os.listdir(full_dir), desc='====processing train database data'):
                        full_path = os.path.join(full_dir, file)
                        img = self.transform(Image.open(full_path).convert('RGB')).to(self.device).unsqueeze(0)
                        embeddings = self.vit_model(img)
                        self.embeddings.append(embeddings[1].detach().cpu().numpy())
                        self.address.append(full_path)
                    self.embeddings = np.array(self.embeddings).reshape(-1,self.embed_dim) 
                    print(f'====train database embeddings shape: {self.embeddings.shape}')
                    self.alto_embed_paths = {'embeddings':self.embeddings,'address':self.address}
                        
                elif dir == 'queries'and dir == self.args.dataset_mode:
                    full_dir = os.path.join(self.dataset_model_path,'queries')
                    for file in tqdm(os.listdir(full_dir), desc='====processing train queries data'):
                        full_path = os.path.join(full_dir, file)
                        img = self.transform(Image.open(full_path).convert('RGB')).to(self.device).unsqueeze(0)
                        embeddings = self.vit_model(img)
                        self.embeddings.append(embeddings[1].detach().cpu().numpy())
                        self.address.append(full_path)
                    self.embeddings = np.array(self.embeddings).reshape(-1,self.embed_dim) 
                    print(f'====train queries embeddings shape: {self.embeddings.shape}')
                    self.alto_embed_paths = {'embeddings':self.embeddings,'address':self.address}
                else:pass
        elif self.args.dataset_proc == 'val':
            # the dataset_model_path(parent dir) with children dirs
            database_query_dirs = os.listdir(self.dataset_model_path)
            for dir in database_query_dirs:
                if dir == 'database'and dir == self.args.dataset_mode:
                    full_dir = os.path.join(self.dataset_model_path,'database')
                    for file in tqdm(os.listdir(full_dir), desc='====processing val database data'):
                        full_path = os.path.join(full_dir, file)
                        img = self.transform(Image.open(full_path).convert('RGB')).to(self.device).unsqueeze(0)
                        embeddings = self.vit_model(img)
                        self.embeddings.append(embeddings[1].detach().cpu().numpy())
                        self.address.append(full_path)
                    self.embeddings = np.array(self.embeddings).reshape(-1,self.embed_dim) 
                    print(f'====val database embeddings shape: {self.embeddings.shape}')
                    self.alto_embed_paths = {'embeddings':self.embeddings,'address':self.address}
                        
                elif dir == 'queries'and dir == self.args.dataset_mode:
                    full_dir = os.path.join(self.dataset_model_path,'queries')
                    for file in tqdm(os.listdir(full_dir), desc='====processing val queries data'):
                        full_path = os.path.join(full_dir, file)
                        img = self.transform(Image.open(full_path).convert('RGB')).to(self.device).unsqueeze(0)
                        embeddings = self.vit_model(img)
                        self.embeddings.append(embeddings[1].detach().cpu().numpy())
                        self.address.append(full_path)
                    self.embeddings = np.array(self.embeddings).reshape(-1,self.embed_dim) 
                    print(f'====val queries embeddings shape: {self.embeddings.shape}')
                    self.alto_embed_paths = {'embeddings':self.embeddings,'address':self.address}
                else:pass
        else:raise ValueError('no more dataset_name satisfied the train/test/val')
        # return the embeddings and adress with format (N, embed_dim)
        return self.alto_embed_paths
    
    def __save_embeddata__(self):
        # save the processed dataset data: embeddings and image_paths
        alto_embed_paths = self.__getitem__()
        self.save_dir = os.path.join(self.args.save_dir, self.args.save_mode)
        self.save_filepath = os.path.join(self.save_dir,self.args.model_name+'_'+self.args.dataset_mode+
                  '_'+self.args.dataset_name+'_'+self.args.dataset_proc+'.h5')
        print(self.save_filepath)
        print(f'====Ready to save the embeddings and the filepath is below:\n{self.save_filepath}')
        self.use_h5py_save_features(alto_embed_paths, self.save_filepath)
        print('====All done!')
    
    def use_h5py_save_features(self, samples_embedings, save_path):
        if isinstance(samples_embedings, np.ndarray) and samples_embedings is not None:
            with h5py.File(save_path, 'w') as h5fw: 
                h5fw.create_dataset("data", data=samples_embedings, dtype='float32')
                print('====features has been saved to "{}"" '.format(save_path))
        elif isinstance(samples_embedings, dict) and samples_embedings is not None:
            with h5py.File(save_path, 'w') as f:
                dict_group = f.create_group('data') 
                for key, value in samples_embedings.items():
                    dict_group.create_dataset(key, data=value)
                print('====features has been saved to "{}" '.format(save_path))
        else: raise ValueError('==== error: The given samples_embedings is not qualified for requirements.')
                
        
        
########################### 主程序测试 #######################################
# python embed_nwpu.py --dataset_name=nwpu --dataset_proc=val_0407  
# --dist_offset=offset_0_None --save_mode=database --dataset_mode=queries
#############################################################################
if __name__ == '__main__':
    # format the alto dataset to matrix and save it
    args = parsers()
    alto_dataset = AltoDataset(args, transform=selavpr_transform)
    alto_embed_paths = alto_dataset.__save_embeddata__()