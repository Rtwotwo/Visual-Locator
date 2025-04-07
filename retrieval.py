"""
时间: 2024/10/18
任务: 使用预训练权重Prithvi_100M.pt权重以及LSHash,构建遥感图像分类算法
    (1) 使用预训练权重提取图像特征向量
    (2) 使用LSHash算法将查询图像与数据库中的图像进行相似度计算
    (3) 输出排序后的与查询图像最相似的10张图像名称
TODO 测试总体的检索+匹配的精度以及IoU占比,
     需要重新编写retrieval.py函数,至少包含两类检索方式
     2025/03/03-Redal
"""
import os, glob
import cv2
import h5py 
from time import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import faiss

import torch
import torch.nn.functional as F
from datetime import datetime
from torchmetrics.retrieval import RetrievalMAP
from utils.hash_code import get_hash
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
import argparse
from models.sift.SIFT import sift_candidate_match
from models.sift.SIFT import sift_match
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def read_h5py_data(h5py_file_path):
    """"用于读取h5py文件的字典数据"""
    if os.path.exists(h5py_file_path):
        with h5py.File(h5py_file_path, 'r') as f:
            dict_group = f['data']
            dict_embedding = {}
            for key in dict_group.keys():
                dict_embedding[key] = dict_group[key][()]
            return dict_embedding
       
def embedding_dict_to_list(embedding_dict, embed_dim=768):
    """将字典形式的embedding转换为列表形式"""
    embedding = embedding_dict['embeddings']
    address = embedding_dict['address']
    return embedding, address 

def get_similarity(queries, database, distance='hamming'):
    """
    计算查询图像queries和数据库图像database之间的相似度
    queries(np.array): 查询图像特征向量组成的数组，形状为[n_queries, n_features]
    database(np.array): 数据库图像特征向量组成的数组，形状为[n_database, n_features]
    distance(str): 距离测量方法，支持'hamming','euclidean'等
    Returns: similarity: 查询图像与数据库图像间的相似度，形状为[n_queries, n_database]"""
    if distance == 'hamming':
        return -torch.cdist(queries.float(), database.float(), p=1)
    elif distance == 'euclidean':
        return -torch.cdist(queries.float(), database.float(), p=2)
    elif distance == 'cosine':
        return F.cosine_similarity(queries.float().unsqueeze(1), database.float().unsqueeze(0), dim=-1)
    elif distance == 'dotproduct':
        return torch.einsum('ab,cb->ac', queries, database).float()
    else:
        raise NotImplementedError(f"The distance {distance} has not been implemented")
    
def find_top_k_images(database_emb_path, queries_emb_path, k, distance='hamming',
                        hash_method='lsh',hash_length=64,hash_threshold=0, hash_seed=42,second_retrieval =None,faiss_k = 5):
    """
    使用queries_embedding和database_embedding来进行查询图像检索实验
    queries_embedding(np.array): 查询图像特征向量组成的数组，形状为[n_queries, n_features]
    database_embedding(np.array): 数据库图像特征向量组成的数组，形状为[n_database, n_features]"""
    database_embedding, database_address = embedding_dict_to_list(read_h5py_data(database_emb_path))
    queries_embedding, _ = embedding_dict_to_list(read_h5py_data(queries_emb_path))
    database_embedding = torch.tensor(database_embedding).to(device)
    queries_embedding = torch.tensor(queries_embedding).to(device)
    # 使用原始embedding测试
    similarity = get_similarity(queries_embedding, database_embedding, distance='hamming')
    similarity = similarity.to(device)
    # 查询前k个索引
    top_k_indices = torch.topk(similarity, k, dim=-1).indices.cpu().numpy()
    #构建最相似的图像的路径列表
    top_k_image_paths = []
    for idxes in top_k_indices:
        paths = [database_address[idx] for idx in idxes]
        top_k_image_paths.append(paths)
    return top_k_image_paths

def check_Recall_x(database_emb_path, queries_emb_path, k):
    """根据查询路径找到返回的"""
    def extract_label(paths_list): 
        label_list = []
        for path in paths_list:
            path = str(path).split('/')[-1]
            label = path.split('_')[0]
            label_list.append(label)
        return label_list
    # 获取queries和top_k_images的label列表
    queries_label = extract_label(queries_emb_path) 
    database_label = [extract_label(paths) for paths in database_emb_path]

    count, all_length = 0, len(queries_label)
    print(f'====all length: {all_length}')
    for i in tqdm(range(all_length),desc=f'====Compute Recall{k}'): 
        if queries_label[i] in database_label[i]:
            count += 1 
    print(f"====Recall@{k}: {count / all_length*100:.4f}")   



if __name__ == '__main__':
    # database_emb_path = 'embedding/database/selavpr_database_nwpu_train.h5'
    # queries_emb_path = 'embedding/queries/selavpr_queries_nwpu_train.h5'
    database_emb_path = '/data2/dataset/Redal/Redal/embedding/database/SelaVPR_database_tianzhibei_train.h5'
    queries_emb_path = '/data2/dataset/Redal/Redal/embedding/queries/SelaVPR_tianzhibei_train.h5'
    current_path = os.getcwd()
    print(current_path)
    # 读取数据
    database_embedding, database_address = embedding_dict_to_list(read_h5py_data(database_emb_path))
    queries_embedding, queries_address = embedding_dict_to_list(read_h5py_data(queries_emb_path))
    database_embedding = torch.tensor(database_embedding).to(device)
    queries_embedding = torch.tensor(queries_embedding).to(device)
    print(f'====database shape:{database_embedding.shape}, queries shape:{queries_embedding.shape}====')
    
    # top_k = 20
    # test_index = 241 #测试第test_index作为查询图像
    # top_k_image_paths = find_top_k_images(database_emb_path, queries_emb_path, k=top_k, hash_length=64,second_retrieval='faiss')
    # print(f'===={test_index} image adress: "{queries_address[test_index]}"')
    # print(f'====retrieval answer:') 
    # for i in top_k_image_paths[test_index]: 
    #     i = os.path.join(current_path, i.decode('utf-8'))
    #     print(f'===={i}')
    # # 进行图像配准
    # query_image_path = os.path.join(current_path, 'Redal', queries_address[test_index].decode('utf-8'))
    # candi_image_paths= [os.path.join(current_path,'Redal',i.decode('utf-8')) for i in top_k_image_paths[test_index]]
    # best_similarity = sift_candidate_match(query_image_path, candi_image_paths, dis_threshold=0.95)
    # print(f'====Best Candidate Image Path:\n {best_similarity}')
    # _, good_match = sift_match(query_image_path, best_similarity, dis_threshold=0.65)
    # cv2.imwrite(os.path.join(current_path, 'output/output.jpg'), good_match)
    # cv2.imwrite(os.path.join(current_path, 'output/query.jpg'), cv2.imread(query_image_path))
    # cv2.imwrite(os.path.join(current_path, 'output/candi.jpg'), cv2.imread(best_similarity))


    # 测试Recall@x的表现
    for top_k in [1, 5, 10, 15, 20, 25]:
        start_time = time()
        top_k_image_paths = find_top_k_images(database_emb_path, queries_emb_path, k=top_k, hash_length=128,second_retrieval='faiss',faiss_k=5)
        check_Recall_x(top_k_image_paths, queries_address, k=top_k)
        end_time = time()
        wasted_time = end_time-start_time;print(f'====Total wasted time:{wasted_time}s\n')