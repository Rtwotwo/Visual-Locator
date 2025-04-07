"""
时间: 2024/10/18
任务: 使用LSHash算法实现Hash检索算法
"""
import torch
import numpy as np
from lshashpy3 import LSHash


def get_hash(embedding, method='trivial', *args, **kwargs):
    """使用定义的方法将嵌入或嵌入列表转换为哈希码"""
    if not isinstance(embedding, torch.Tensor):
        if isinstance(embedding, list):
            # 递归调用,遍历嵌入列表
            return [get_hash(e, method, *args, **kwargs) for e in embedding]
        else: 
            raise TypeError('==== error: embedding must be tensor or list of tensors')
        
    if method == 'trivial': return trivial_hash(embedding, *args, **kwargs)
    elif method == 'lsh': return lshash(embedding, *args, **kwargs)
    elif method == 'none': return embedding
    else: raise NotImplementedError('==== error: ' + method + ' hash has not been implemented') 
    

def trivial_hash(embedding: torch.Tensor, length: str =64, threshold: float=0., seed=None):
    """
    通过平均多个嵌入维度并使用阈值进行二值化来创建一个简单的二进制哈希
    """
    assert embedding.size(-1) % length == 0, \
        f"Cannot create hash with length {length} with embedding dim {embedding.size(-1)}"
    resize_factor = int(embedding.size(-1) / length)
    binary_hash = embedding.reshape([-1, resize_factor, length]).mean(dim=1) > threshold
    return binary_hash.int()


def lshash(embedding: torch.Tensor, length: str = 64, store: str = None, seed: int = 42):
    """通过从论文中应用 LSH 创建二进制哈希：
    用于在概率分布中查找最近邻居的局部敏感哈希.
    Using the implementation from https://github.com/loretoparisi/lshash."""
    np.random.seed(seed) # 初始化局部敏感哈希
    lsh = LSHash(hash_size=length, input_dim=embedding.size(-1), num_hashtables=1,
                 hashtable_filename=store, overwrite=True)
    # 为每个嵌入生成哈希
    hashes = []
    for e in embedding:
        h = lsh.index(e.tolist(), extra_data=None)
        hashes.append(list(map(int, h[0])))
    return torch.Tensor(hashes).int()