# ECEN649 FInal Project
# Viola Jones face detection
# Lipai Huang and Inuk Kang

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from func.log import kill_empty_log, Log
from tqdm import tqdm
from func.data_loader import load_data
from func.haar_feature import haar_features_main
from func.evaluate import evaluate, draw_ROC, add_summary
from func.adaboost import adaboost, adaboost_validate, adaboost_predict

# init params
# set all the image shape to be 19 x 19
img_shape = (19, 19)
# store and evaluate the prediction info for specific rounds
rounds = [1, 3, 5, 10]
# set the basic haar kernel shape
basic_harr_list = [[1, 2], [2, 1], [1, 3], [3, 1], [2, 2]]

# skip generating haar features if the .npy files are already exist
def check_cache():
    cache_dir = 'cache'
    train_face_cache_path = os.path.join(cache_dir, 'train_faces_haar.npy')
    train_non_face_cache_path = os.path.join(cache_dir, 'train_non_faces_haar.npy')
    if os.path.exists(train_face_cache_path) and os.path.exists(train_non_face_cache_path):
        train_face_haar = np.load(train_face_cache_path)
        train_non_face_haar = np.load(train_non_face_cache_path)
        return train_face_haar, train_non_face_haar
    else:
        return None, None

# main func
if __name__ == '__main__':
    # load data from files and check
    print('-'*120)
    print('Loading data...')
    load_log = Log('load')
    try:
        face_train, non_face_train = load_data('datasets\\trainset', img_shape)
        face_test, non_face_test = load_data('datasets\\testset', img_shape, num_face=4000)
    except:
        load_log.exception()
    print(face_train.shape, non_face_train.shape)
    print(face_test.shape, non_face_test.shape)
    print('Data loaded!')
    print('-'*120)
    
    # generate or load haar features and check
    haar_log = Log('haar')
    try:
        train_face_features, train_non_face_features = check_cache()
        if type(train_face_features)==type(None) or type(train_non_face_features)==type(None):
            print('Generating face Haar Features, it may take a long wait...')
            train_face_features = haar_features_main(face_train, img_shape, basic_harr_list)
            print('Generating non face Haar Features, it may take a long wait...')
            train_non_face_features = haar_features_main(non_face_train, img_shape, basic_harr_list)
            train_face_cache_path = 'cache\\train_faces_haar.npy'
            open(train_face_cache_path, 'a').close()
            np.save(train_face_cache_path, train_face_features)
            train_non_face_cache_path = 'cache\\train_non_faces_haar.npy'
            open(train_non_face_cache_path, 'a').close()
            np.save(train_non_face_cache_path, train_non_face_features)
            print('Haar features generated!')
        else:
            print('Features loaded from cache!')
    except:
        haar_log.exception()
    # form train and test arrays
    X_train = np.append(face_train, non_face_train, axis=0)
    X_train_features = np.append(train_face_features, train_non_face_features, axis=0)
    y_train = np.append(np.ones(face_train.shape[0]), -1*np.ones(non_face_train.shape[0]), axis=0)
    X_test = np.append(face_test, non_face_test, axis=0)
    y_test = np.append(np.ones(face_test.shape[0]), -1*np.ones(non_face_test.shape[0]), axis=0)
    
    print('-'*120)
    
    print('Training Adaboost...')
    ada_log = Log('ada')
    # load kernels information
    kernels_info = np.load('cache\kernels.npy').reshape(-1, 5)
    # .txt file for summary
    with open('outputs\\Adaboost_report.txt', 'w+') as f:
        # ROC curve contains specific rounds
        fig = plt.figure(0)
        # training
        try:
            clf_list, weight_list, j_list = adaboost(X_train_features, y_train, rounds)
            # print(clf_list)
            # print(weight_list)
            # start iteration
            print('-'*120)
            print('Predicting and evaluating...')
            for idx in tqdm(range(len(rounds))):
                t = rounds[idx]
                # validation
                y_val = adaboost_validate(X_train_features, clf_list[:t], weight_list[t-1])
                # y_val = adaboost_predict(X_train, img_shape, clf_list, weight_list, kernels_info)
                train_TPR, train_FPR, train_accuracy = evaluate(y_val, y_train)
                # prediction
                y_pred = adaboost_predict(X_test, img_shape, clf_list[:t], weight_list[t-1], kernels_info)
                test_TPR, test_FPR, test_accuracy = evaluate(y_pred, y_test)
                # write prediction evaluation for specific rounds into .txt file
                f.write(f'Adaboost round set {t}:\n')
                f.write(f'Training summary: TPR={train_TPR}, FPR={train_FPR}, accuracy={train_accuracy}\n')
                f.write(f'Test summary: TPR={test_TPR}, FPR={test_FPR}, accuracy={test_accuracy}\n')
                table, combined_clf = add_summary(clf_list[:t], weight_list[t-1], j_list[:t], kernels_info)
                f.write(table+'\n')
                f.write(combined_clf)
                draw_ROC(X_test, y_test, img_shape, clf_list[:t], weight_list[t-1], t, kernels_info)
                f.write('-'*120)
                # embed clf into img, not completed!
                if t == max(rounds):
                    cfg_path = 'cache\\best_config'
                    if not os.path.exists(cfg_path):
                        os.mkdir(cfg_path)
                    np.save(os.path.join(cfg_path, 'clf_list.npy'), clf_list)
                    np.save(os.path.join(cfg_path, 'weight_list.npy'), weight_list[t-1])
                    sample_face = X_test[y_pred + y_test == 2][0]
            fig.savefig('outputs\\ROC_curve.png', dpi=fig.dpi)
        except:
            ada_log.exception()
    print('-'*120)
    print('All processes done! Check outputs!')
    

