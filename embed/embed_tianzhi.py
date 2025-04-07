"""
任务: 处理数据集,生成queries和database的embedding存储在本地
      使用Prithvi或者SelaVPR模型来提取特征进行演示
时间: 2024/10/18-Redal      
"""
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
from Redal.models.prithvi.Prithvi_ViT import RetrievalViT 
from models.selavpr import network
from functools import partial
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms 
from PIL import Image
import h5py 


# 图片进行预transforms处理 
Prithvi_dataset_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
SelaVPR_dataset_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])


class DatasetNet(Dataset):
    def __init__(self,embed_dim=768,Datasets_Path=None, 
                 dataset_name=None, mode = 'train',
                 transform=None, model=None, model_selection=None):
        super().__init__()
        queries_dirpath = os.path.join(Datasets_Path, dataset_name, mode, 'queries')
        dataset_dirpath = os.path.join(Datasets_Path, dataset_name, mode, 'database')
        self.dataset_name = dataset_name
        self.mode = mode
        self.model = model
        self.embed_dim = embed_dim
        self.model_selection = model_selection
        self.queries_filepath_list = [os.path.join(queries_dirpath,path) for path in os.listdir(queries_dirpath)]
        self.database_filepath_list = [os.path.join(dataset_dirpath,path) for path in os.listdir(dataset_dirpath)]
        self.transform = transform
        self.chan_proj = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=1,) #扩增为6通道
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    def __len__(self):
        return (len(self.queries_filepath_list), len(self.database_filepath_list))
    
    def compute_database(self):
        """
        计算database数据库的embedding
        output: (samples, embeddings),  embeddings -> 768/1024
        """
        if isinstance(self.model, nn.Module):
            self.model.eval().to(self.device)
            if self.model_selection == 'Prithvi':
                with torch.no_grad():
                    database, database_embedding = {}, [] #使用dict保存数据(地址-list,embedding-ndarray): {图像地址1: embedding1, 图像地址2: embedding2,...}
                    for filepath in tqdm(self.database_filepath_list, desc=f'====Compute {self.mode} database embedding'):
                        if self.transform :
                            img = self.transform(Image.open(filepath))
                            img = self.chan_proj(img).unsqueeze(1).unsqueeze(0).to(self.device)
                            
                        embedding = self.model(img).cpu().numpy() # (1, embed_dim)
                        database_embedding.append(embedding)
                database_embedding = np.array(database_embedding).reshape(-1,self.embed_dim)
                database = {'embeddings':database_embedding, 'address':self.database_filepath_list}
                return database
            elif self.model_selection =='SelaVPR':
                with torch.no_grad():
                    database, database_embedding = {}, [] #使用dict保存数据(地址-list,embedding-ndarray): {图像地址1: embedding1, 图像地址2: embedding2,...}
                    for filepath in tqdm(self.database_filepath_list, desc=f'====Compute {self.mode} database embedding'):
                        if self.transform :
                            img = self.transform(Image.open(filepath))
                            img = img.unsqueeze(0).to(self.device)
                        embedding = self.model(img) # (1, embed_dim)
                        database_embedding.append(embedding[1].cpu().numpy())
                database_embedding = np.array(database_embedding).reshape(-1,self.embed_dim)
                database = {'embeddings':database_embedding, 'address':self.database_filepath_list}
                return database
            else:
                raise ValueError('====error there is no needed model ====')     
        else: raise ValueError('==== error: The given model is None.')

    def compute_queries(self):
        """
        计算queries数据库的embedding
        output: (samples, embeddings),  embeddings -> 768 
        """
        if isinstance(self.model, nn.Module):
            self.model.eval().to(self.device)
            if self.model_selection == 'Prithvi':
                with torch.no_grad():
                    queries, queries_embedding = {}, [] #使用dict保存数据
                    for filepath in tqdm(self.queries_filepath_list, desc=f'====Compute {self.mode} queries embedding'):
                        if self.transform is not None:
                            img = self.transform(Image.open(filepath))
                            img = self.chan_proj(img).unsqueeze(1).unsqueeze(0).to(self.device)
                        embedding = self.model(img).cpu().numpy() # (1, embed_dim)
                        queries_embedding.append(embedding) #torch.tensor(path)   #TODO 计算数据库的embedding
                queries_embedding = np.array(queries_embedding).reshape(-1,self.embed_dim) 
                print(queries_embedding.shape)
                queries = {'embeddings':queries_embedding,'address':self.queries_filepath_list}
                return queries
            
            elif self.model_selection =='SelaVPR':
                with torch.no_grad():
                    queries, queries_embedding = {}, [] #使用dict保存数据
                    for filepath in tqdm(self.queries_filepath_list, desc=f'====Compute {self.mode} queries embedding'):
                        if self.transform is not None:
                            img = self.transform(Image.open(filepath))
                            img = img.unsqueeze(0).to(self.device)
                        embedding = self.model(img) # (1, embed_dim)
                        queries_embedding.append(embedding[1].cpu().numpy()) #torch.tensor(path)   #TODO 计算数据库的embedding
                queries_embedding = np.array(queries_embedding).reshape(-1,self.embed_dim) 
                print(queries_embedding.shape)
                queries = {'embeddings':queries_embedding,'address':self.queries_filepath_list}
                return queries
            else:
                raise ValueError('====error there is no needed model ====')
        else: raise ValueError('==== error: The given model is None.')

def get_pretrained_model(pretrained_model_path, model=None):
    """
    获取预训练的模型,导入到本地模型中
    """
    if model=='Prithvi':
        vit_model = RetrievalViT(img_size=224, patch_size=16,
                    num_frames=1, tubelet_size=1,
                    in_chans=6, embed_dim=768, depth=12, num_heads=12,
                    mlp_ratio=4., norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                    norm_pix_loss=False, weights=pretrained_model_path)
    elif model=='SelaVPR':
        vit_model = network.GeoLocalizationNet()
    else:
        raise ValueError('==== error: there is no needed model! ====')
    return vit_model

def use_h5py_save_features(samples_embedings, save_path):
    if isinstance(samples_embedings, np.ndarray) and samples_embedings is not None:
        with h5py.File(save_path, 'w') as h5fw: 
            h5fw.create_dataset("data", data=samples_embedings, dtype='float32')
            print('==== Features has been saved to "{}"" '.format(save_path))
    elif isinstance(samples_embedings, dict) and samples_embedings is not None:
        with h5py.File(save_path, 'w') as f:
            dict_group = f.create_group('data') 
            for key, value in samples_embedings.items():
                dict_group.create_dataset(key, data=value)
            print('==== Features has been saved to "{}" '.format(save_path))
    else: raise ValueError('==== error: The given samples_embedings is not qualified for requirements.') 

def main(model_selection=None):
    for mode in ['val','train','test']:
        # 参数路径定义
        Prithvi_100M_filepath = 'weights/Prithvi_100M.pt'
        SelaVPR_weight_filepath = 'weights/dinov2_vitl14_pretrain.pth'
        Datasets_Path = 'datasets_vg/datasets'
        datasets_name= 'tianzhibei'
        # 定义保存queries和database的features_embedding路径
        database_save_path = f'embedding/database/{model_selection}_database_{datasets_name}_{mode}.h5'
        queries_save_path = f'embedding/queries/{model_selection}_{datasets_name}_{mode}.h5'
    
        if model_selection=='Prithvi':
            vit_model = get_pretrained_model(Prithvi_100M_filepath, model=model_selection)
            features_extract_model = DatasetNet(embed_dim=768, 
                                    Datasets_Path=Datasets_Path, 
                                    dataset_name=datasets_name, mode = mode,
                                    transform=Prithvi_dataset_transforms,
                                    model=vit_model, model_selection=model_selection) 
        elif model_selection=='SelaVPR':
            vit_model = get_pretrained_model(SelaVPR_weight_filepath, model=model_selection)
            features_extract_model = DatasetNet(embed_dim=1024, 
                                    Datasets_Path=Datasets_Path, 
                                    dataset_name=datasets_name, mode = mode,
                                    transform=SelaVPR_dataset_transforms,
                                    model=vit_model, model_selection=model_selection)
        # 计算queries 和 database 的特征数据集并保存
        database_embedding = features_extract_model.compute_database()
        queries_embedding = features_extract_model.compute_queries()
        use_h5py_save_features(database_embedding, database_save_path)
        use_h5py_save_features(queries_embedding, queries_save_path)


if __name__ == '__main__':
    main(model_selection='Prithvi')
    main(model_selection='SelaVPR')