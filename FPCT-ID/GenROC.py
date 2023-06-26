#single evaluation

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from multiprocessing import Pool
import itertools

def calculate_roc(thresholds,
                  p,
                  g,
                  pca=0):
    nrof_thresholds = len(thresholds)
    pool_label = p[:,0]
    pool_data = p[:,1:]
    gallery_label = g[:,0]
    gallery_data = g[:,1:]

    tprs = np.zeros((g.shape[0], nrof_thresholds))
    fprs = np.zeros((g.shape[0], nrof_thresholds))
    
    if pca>0:
        pca_model = PCA(n_components=pca)
        pca_model.fit(pool_data)
        pool_data = pca_model.transform(pool_data)
        gallery_data = pca_model.transform(gallery_data)
    
    pool_data = normalize(pool_data,axis=1)
    gallery_data = normalize(gallery_data,axis=1)
    
    for idx, probe in enumerate(gallery_data):
        diff = np.subtract(pool_data, probe)
        dist = np.sum(np.square(diff), 1)
  
        # Find the best threshold for the fold
        acc_probe = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[idx,threshold_idx], fprs[idx,threshold_idx], acc_probe[threshold_idx] = calculate_accuracy(
                threshold, dist, pool_label==gallery_label[idx])


    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    auc = calculate_auc(fpr, tpr)
    return tpr, fpr, auc


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def evaluate(pool, gallery, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, auc = calculate_roc(thresholds,pool,gallery,pca=pca)
    return tpr, fpr, auc


def cosine_distance(v1, v2):
    """Calculate the cosine distance between two vectors"""
    dot = np.dot(v1, v2)[:,0]
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    cosine_similarity = dot / (norm1 * norm2)
    # return cosine_similarity
    return dot 

def calculate_auc(fpr, tpr):
    auc = 0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
    return auc

def main(combination):
    id, TrainingCRT, TrainingSetInferenceCRT, TestSetInfernceCRT = combination
    pca = 0  
    TrainSet = pd.read_csv(f'TrainingSetInferResults.csv')[['label']+[str(i) for i in range(256)]].to_numpy()
    TestSet = pd.read_csv(f'.TestSetInferResults.csv')[['label']+[str(i) for i in range(256)]].to_numpy()
    tpr, fpr, auc = evaluate(TrainSet, TestSet, pca)
    summary = np.array([fpr,tpr]).T
    pd.DataFrame(summary, columns=['FPR','TPR']).to_csv(f'ROCResults.csv',index=False)
    
if __name__ == '__main__':
    pool = Pool()
    id_list = ['v','vc','vnc','vn']
    TF_list = [True, False]
    combinations = itertools.product(id_list,TF_list, TF_list,TF_list)
    pool.map(main,combinations)
    pool.close()
    pool.join()