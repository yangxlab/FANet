import argparse
import scipy.io
import torch
import numpy as np
import os
import os.path as osp
from torchvision import datasets
import matplotlib
from PIL import Image
matplotlib.use('agg')
import matplotlib.pyplot as plt
from data_manager import VCM
# import cide
# from tools.video_loader import VideoDataset
#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--index', default=500, type=int, help='test_image_index')
opts = parser.parse_args()

dataset = VCM()

#####################################################################
#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    im = Image.fromarray(im)
    im = im.resize((144, 288))
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
t2v = True
######################################################################
if t2v:
    # result = scipy.io.loadmat('/media/yangxilab/DiskB/wxian/VI-Video-Baseline1/FANet_t2v_results.mat')
    result = scipy.io.loadmat('/media/yangxilab/DiskB/wxian/MITML-main/Base_t2v_results.mat')
    query_feature = torch.FloatTensor(result['IRq_f'])
    query_cam = result['IRq_camids'][0]
    query_label = result['IRq_pids'][0]
    gallery_feature = torch.FloatTensor(result['RGBg_f'])
    gallery_cam = result['RGBg_camids'][0]
    gallery_label = result['RGBg_pids'][0]
else:
    # result = scipy.io.loadmat('/media/yangxilab/DiskB/wxian/VI-Video-Baseline1/FANet_v2t_results.mat')
    result = scipy.io.loadmat('/media/yangxilab/DiskB/wxian/MITML-main/Base_v2t_results.mat')
    query_feature = torch.FloatTensor(result['RGBq_f'])
    query_cam = result['RGBq_camids'][0]
    query_label = result['RGBq_pids'][0]
    gallery_feature = torch.FloatTensor(result['IRg_f'])
    gallery_cam = result['IRg_camids'][0]
    gallery_label = result['IRg_pids'][0]
query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()
print(query_feature.shape,query_label.shape,query_cam.shape,gallery_feature.shape,gallery_label.shape,gallery_cam.shape)  # 1980 9330
#######################################################################
# sort the images
def sort_img(qf, ql, qc, gf, gl, gc):

    qf  = qf.view(1,2048) 
    # gf = gf[1980:,:]
    print(qf.shape,gf.shape)
    m, n = qf.size(0), gf.size(0)
    q_g_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    q_g_dist.addmm_(1, -2, qf, gf.t())
    q_g_dist = q_g_dist.reshape(gf.size(0))
    score = q_g_dist.cpu().numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    # good index
    query_index = np.argwhere(gl==ql)
    #same camera
    camera_index = np.argwhere(gc==qc)

    #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) 

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index


########################################################################
# Visualize the rank result
def get_rank_result(dataset, index):
    i = index
    if t2v:
        root = osp.join("./rank/","t2v")
    else :
        root = osp.join("./rank/","v2t")
    index = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
    print(i,query_feature[i].shape,query_label[i],query_cam[i])
    print(index[0:5])
    print("------query---------")
    if t2v :
        query_path, qpid, _= dataset.query[i]
    else:
        query_path, qpid, _= dataset.query_1[i]

    print(query_path[0],query_path[-1])


    print("------gallery-------")
    for j in range(10):
        if t2v :
            gallery_path, gpid, _ = dataset.gallery[index[j]]
        else:
            gallery_path, gpid, _= dataset.gallery_1[index[j]]
        print("top"+str(j+1)+":" + gallery_path[0]+"___"+gallery_path[-1],gpid)
        
    print('Top 10 images are as follow:')
    try: # Visualize Ranking Result 
        # Graphical User Interface is needed
        fig = plt.figure(figsize=(16,4))
        ax = plt.subplot(1,11,1)
        ax.axis('off')
        query_path=list(query_path)
        imshow(query_path[0],'query')
        for i in range(10):
            ax = plt.subplot(1,11,i+2)
            ax.axis('off')
            if t2v :
                img_path, gpid, _ = dataset.gallery[index[i]]
            else:
                img_path, gpid, _= dataset.gallery_1[index[i]]
            img_path=list(img_path)
            imshow(img_path[0])
            if gpid == qpid:
                ax.set_title('%d'%(i+1), color='green')
            else:
                ax.set_title('%d'%(i+1), color='red')
    except RuntimeError:
        print('If you want to see the visualization of the ranking result, graphical user interface is needed.')
    
    fig.savefig(osp.join(root,f"{qpid}_Base.png"))

for i in range(opts.index):
    if i % 15 == 0:
        get_rank_result(dataset, i)