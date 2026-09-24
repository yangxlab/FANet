import os
import random
import numpy as np
import scipy.io as sio
import matplotlib as mpl

mpl.use('AGG')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_embedding(resultq, resultg, query_id, gallery_id, title):
    print('resultq',resultq.shape)
    data = np.concatenate((resultq, resultg),axis=0)
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    print(x_min,x_max)
    # x_min=x_min-40
    # x_max=x_max+40
    print(x_min,x_max)
    # resultq = (resultq - x_min) / (x_max - x_min)
    # resultg = (resultg - x_min) / (x_max - x_min)
    resultq = (resultq - x_min)*100
    resultg = (resultg - x_min)*100
    fig = plt.figure()
    ax = plt.subplot(111)
    qid=list(set(query_id))
    print('qid',qid,query_id,resultq.shape)
    gid=list(set(gallery_id))
    print('gid',gid,gallery_id,resultg.shape)
    color=['r','orange','gray','yellow','g','cyan','b','violet','pink','k']
    for i in range(resultq.shape[0]):
        print(resultq[i, 0], resultq[i, 1])
        plt.text(resultq[i, 0], resultq[i, 1], '*',
			 color=color[qid.index(query_id[i])] ,
			 fontdict={'weight': 'bold', 'size':9})
    for i in range(resultg.shape[0]):
        print(resultg[i, 0], resultg[i, 1])
        plt.text(resultg[i, 0],resultg[i, 1], '.',
			 color=color[gid.index(gallery_id[i])],
			 fontdict={'weight': 'bold', 'size':12})
    # plt.xticks([])
    # plt.yticks([])
    plt.xlim(0,20)
    plt.ylim(0,20)
    plt.title(title)
    return fig




if __name__ == '__main__':
    test_ids = [
        6, 10, 17, 21, 24, 25, 27, 28, 31, 34, 36, 37, 40, 41, 42, 43, 44, 45, 49, 50, 51, 54, 63, 69, 75, 80, 81, 82,
        83, 84, 85, 86, 87, 88, 89, 90, 93, 102, 104, 105, 106, 108, 112, 116, 117, 122, 125, 129, 130, 134, 138, 139,
        150, 152, 162, 166, 167, 170, 172, 176, 185, 190, 192, 202, 204, 207, 210, 215, 223, 229, 232, 237, 252, 253,
        257, 259, 263, 266, 269, 272, 273, 274, 275, 282, 285, 291, 300, 301, 302, 303, 307, 312, 315, 318, 331, 333
    ]
    for i in range(len(test_ids)):
        test_ids[i] += 500
        
    random.seed(0)
    tsne = TSNE(n_components=2, init='pca')
    selected_ids = random.sample(test_ids, 20)
    # plt.figure(figsize=(5, 5))
    print(selected_ids)
    v2t = False
    # features without dual path
    if v2t:
        mat_path = '/media/yangxilab/DiskB/wxian/VI-Video-Baseline1/FANet_v2t_results.mat'

        mat = sio.loadmat(mat_path)
        q_feats = mat["RGBq_f"]
        q_ids = mat["RGBq_pids"].squeeze()
        # print(q_feats,q_ids)
        flag = np.in1d(q_ids, selected_ids)
        q_ids = q_ids[flag]
        q_feats = q_feats[flag]

        g_feats = mat["IRg_f"]
        g_ids = mat["IRg_pids"].squeeze()
        # print(g_feats,g_ids)
        flag = np.in1d(g_ids, selected_ids)
        g_ids = g_ids[flag]
        g_feats = g_feats[flag]
    else:
        # mat_path = '/media/yangxilab/DiskB/wxian/VI-Video-Baseline1/FANet_t2v_results.mat'
        mat_path = '/media/yangxilab/DiskB/wxian/MITML-main/Base_t2v_results.mat'
        mat = sio.loadmat(mat_path)
        q_feats = mat["IRq_f"]
        q_ids = mat["IRq_pids"].squeeze()
        flag = np.in1d(q_ids, selected_ids)
        # print(flag)
        q_ids = q_ids[flag]
        q_feats = q_feats[flag]

        g_feats = mat["RGBg_f"]
        g_ids = mat["RGBg_pids"].squeeze()
        # print(g_ids.shape)
        flag = np.in1d(g_ids, selected_ids)
        # print(flag.shape)
        g_ids = g_ids[flag]
        g_feats = g_feats[flag]

    print(q_feats.shape,g_feats.shape,q_ids.shape,g_ids.shape)
    feat = np.concatenate([q_feats, g_feats], axis=0)
    embed = tsne.fit_transform(feat)
    q_samples = q_feats.shape[0]

    resultq=feat[0:q_samples]
    resultg=feat[q_samples:]

    # fig = plot_embedding(resultq, resultg, q_ids, g_ids,'t-SNE embedding of the digits')
    # plt.savefig('./bicenter.png')
    # colors=['r','orange','gray','yellow','g','cyan','b','violet','pink','k']

    # c = [colors[i % len(colors)] for i in range(q_feats.shape[0] + g_feats.shape[0])]

    c = ['r'] * q_feats.shape[0] + ['g'] * g_feats.shape[0]

    # plt.subplot(1, 2, 1)
    plt.scatter(embed[:, 0], embed[:, 1], c=c)

    plt.tight_layout()
    plt.savefig('tsne.jpg')


