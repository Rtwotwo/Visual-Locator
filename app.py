"""
任务: 测试总体的检索+匹配的精度以及IoU占比,
      需要重新编写retrieval.py函数,至少包含两类检索方式
TODO: set a matches threashold at 50
时间: 2025/03/03-Redal
"""
import os
import h5py
import argparse
import torch
import cv2
from PIL import Image
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from torchvision.transforms import transforms
# from torchmetrics.retrieval import RetrievalMAP
from utils.hash_code import get_hash
from models.selavpr import network
from models.sift.SIFT import sift_candidate_match
from models.sift.SIFT import sift_match, sift_match_pro
from models.spsg.spsg import superpoint_match, superpoint_match_pro, superglue_match

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
selavpr_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



######################### 定义变量解析阈  #################################
def config():
    parser = argparse.ArgumentParser(description='parameters for retrieval',
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--embed_dir', type=str, default='/data2/dataset/Redal/Redal/embedding',
                        help='storing the embedded data dir and consisting of database and queries')
    parser.add_argument('--save_visual', type=str, default='/data2/dataset/Redal/Redal/output',
                        help='save the candidate, query and registration into output dir')
    parser.add_argument('--embed_class', type=str, default='nwpu',
                        help='embed_class consists of nwpu/UAV/tianzhibei')
    parser.add_argument('--method', type=str, default='sift',
                        help='method for retrievaling the correct image consist of sift/spsg/loftr')
    parser.add_argument('--top_k', type=int, default=25, 
                        help='find the k images which satisfing the query')
    parser.add_argument('--simi_method', type=str, default='hamming',
                        help='some methods are hamming/euclidean/cosine/dotproduct/')
    parser.add_argument('--is_recall', type=bool, default=False,
                        help='whether retrievaling for recall@x testing or not')
    parser.add_argument('--single_query', type=str, default=None,
                        help='choose a query image path for testing')    
    parser.add_argument('--weight_dir', type=str, default='/data2/dataset/Redal/Redal/weights',
                        help='please write down the correct weight direction')
    parser.add_argument('--weight_name', type=str, default='dinov2_vitl14_pretrain.pth',
                        help="there are some choices ['dinov2_vitl14_pretrain.pth', 'Prithvi_100M.pt']")
    args = parser.parse_args()
    return args



##########################  定义检索匹配函数  ########################################
def read_data(h5py_file_path):
    """"read the embedding and adress in the h5py file"""
    if os.path.exists(h5py_file_path):
        with h5py.File(h5py_file_path, 'r') as f:
            dict_group = f['data']
            dict_embedding = {}
            for key in dict_group.keys():
                dict_embedding[key] = dict_group[key][()]
            return dict_embedding
       
def embed_to_list(embedding_dict):
    """split the data into embedding and adress"""
    embedding = embedding_dict['embeddings']
    address = embedding_dict['address']
    return embedding, address 

def get_similarity(database, queries, distance='hamming'):
    """make sure the retrieval method to find the top_k
    :param database: original database embedding
    :param queries: the uav's frame for retrieval"""
    if distance == 'hamming':       return -torch.cdist(queries.float(), database.float(), p=1)
    elif distance == 'euclidean':   return -torch.cdist(queries.float(), database.float(), p=2)
    elif distance == 'cosine':      return F.cosine_similarity(queries.float().unsqueeze(1), 
                                            database.float().unsqueeze(0), dim=-1)
    elif distance == 'dotproduct':  return torch.einsum('ab,cb->ac', queries, database).float()
    else:raise NotImplementedError(f"The distance {distance} has not been implemented")

def topk_retrieval(args):
    """find the most suitable k candidate images to test
    :param top_k: the number of retried k candidate images"""
    # db_path = os.path.join(args.embed_dir, 'database', f'selavpr_database_{args.embed_class}_train.h5')
    # qu_path = os.path.join(args.embed_dir, 'queries', f'selavpr_queries_{args.embed_class}_train.h5')
    db_path = 'embedding/database/selavpr_references_nwpu_val_0407.h5'
    qu_path = 'embedding/queries/selavpr_queries_nwpu_val_0407.h5'
    db_em, db_ad = embed_to_list(read_data(db_path))
    qu_em, qu_ad = embed_to_list(read_data(qu_path))
    db_em, qu_em = torch.tensor(db_em), torch.tensor(qu_em)
    # get the similarity standard and find similarity index
    similarity = get_similarity(db_em, qu_em, distance=args.simi_method).to(device)
    top_k_indices = torch.topk(similarity, args.top_k, dim=-1).indices.cpu().numpy()
    top_k_paths = []
    for idx in top_k_indices:
        paths = [db_ad[id] for id in idx]
        top_k_paths.append(paths)
    return top_k_paths
    
class VisPosNet:
    """Connect two-stage retrieval and registration models for visualization
    just for only single query image retrieval"""
    def __init__(self, args, **kwargs):
        self.args = args
        self.top_k_paths = None
        self.query_path = args.single_query
        # read the database 
        self.db_path = os.path.join(args.embed_dir, 'database',
                    f'selavpr_database_{args.embed_class}_train.h5')
        self.db_em, self.db_ad = embed_to_list(read_data(self.db_path))
        self.db_em = torch.tensor(self.db_em)
        # initialize the retrieval model
        self.weight_path = os.path.join(args.weight_dir, args.weight_name)
        self.selavpr = network.GeoLocalizationNet(self.weight_path).to(device)
        self.transform = selavpr_transform
        self.query_emb, self.top_k_paths = [], []
        
    def __retrieval__(self,):
        qu_img = Image.open(self.query_path).convert('RGB')
        qu_img = self.transform(qu_img).to(device).unsqueeze(0)
        emb = self.selavpr(qu_img)
        self.query_emb.append(emb[1].detach().cpu().numpy())
        self.query_emb = torch.tensor( np.array(self.query_emb).reshape(-1,1024) )
        # convert bit-class to utf-8 
        similarity = get_similarity(self.db_em, self.query_emb, distance=args.simi_method).to(device)
        self.top_k_indices = torch.topk(similarity, args.top_k, dim=-1).indices.cpu().numpy()
        for idx in self.top_k_indices:
            self.top_k_paths.append( self.db_ad[idx])
        self.top_k_paths = [path.decode('utf-8') for path in self.top_k_paths[0]]
        
    def __registration__(self,):
        self.best_reg = sift_candidate_match(self.query_path, self.top_k_paths, dis_threshold=0.75)
        # find the best_reg path and print current time
        current_time = datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        print(f'{formatted_time}\t best registrated image path:{self.best_reg}')
        
        _, good_match = sift_match(self.query_path, self.best_reg, dis_threshold=0.65)
        cv2.imwrite(os.path.join(self.args.save_visual, 'output.jpg'), good_match)
        cv2.imwrite(os.path.join(self.args.save_visual, 'query.jpg'), cv2.imread(self.query_path))
        cv2.imwrite(os.path.join(self.args.save_visual, 'candi.jpg'), cv2.imread(self.best_reg))
        
class NwpuNet:
    """processing the database and queries embedding data, make a query about 
       the Recall@x, and use the sift/spsg description to rank the candidates """
    def __init__(self,args, **kwargs):
        self.args = args
        self.current_time = datetime.now()
        self.current_time.strftime('%Y-%m-%d %H:%M:%S')
        
    def __retrieval__(self):
        self.top_k_paths = topk_retrieval(self.args)
    def __recall_depend__(self):
        """the number of sift/spsg"""
        # qu_path = os.path.join(self.args.embed_dir, 'queries', 
        #         f'selavpr_queries_{self.args.embed_class}_train.h5')
        qu_path = 'embedding/queries/selavpr_queries_nwpu_val_0407.h5'
        _, qu_ad = embed_to_list(read_data(qu_path))
        qu_ad = [p.decode('utf-8') for p in qu_ad]
        matches_num, count_num = 0, 0
        for idx, candi_path in enumerate(self.top_k_paths):
            # convert bit-class to utf-8 class
            candi_path = [p.decode('utf-8') for p in candi_path]
            candi_best_path = sift_candidate_match(qu_ad[idx], candi_path)
            matches, registration = superpoint_match(qu_ad[idx], candi_best_path)
            # save registrated img into output dir
            if not os.path.exists(self.args.save_visual):
                os.makedirs(self.args.save_visual)
            cv2.imwrite(os.path.join(self.args.save_visual, 'supo',f'output_{matches_num:06d}.jpg'),registration)
            print(f'匹配关键点数目:\t{len(matches)}\t匹配结果路径:\toutput_{matches_num:06d}.jpg')
            matches_num += 1
            if len(matches) > 20:
                count_num += 1
        print(f'检索到的图片数量:\t{matches_num}\t匹配成功的图片数量:\t{count_num}')
        print(f'匹配成功度:{matches_num/count_num*100:.4f}%')

        
###########################  主函数测试  ####################################
""" python app.py --single_query=/data2/dataset/Redal/Redal/datasets_vg
 /datasets/nwpu/train/queries/small_0_8.tif"""
############################################################################
if __name__ == '__main__':
    args = config()
    # if you'll try a single retrieval, below
    # vpn = VisPosNet(args)
    # vpn.__retrieval__()
    # vpn.__registration__()
    # if you want to test embedding data 
    nn = NwpuNet(args)
    nn.__retrieval__()
    nn.__recall_depend__()