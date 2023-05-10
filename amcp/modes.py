
import os
import numpy as np
import joblib
import pandas as pd
import torch
import bz2

from sklearn.model_selection import train_test_split

from . import amcp_lib
from . import ml_models
from . import parsers

from catboost import Pool

import logging
logger = logging.getLogger(__name__)


def train(args):
    """Train ml_models and stores them as compressed files in the models_dir directory along with the calibration conformity scores.

    Args:
        args (_type_): User arguments
    """

    logger.info('Starting training')

    models_dir = args.models_dir

    os.makedirs(models_dir, exist_ok=True)
    logger.info(f'Created the directory: {models_dir}')

    if args.classifier=='BERT':

        data = pd.read_csv(args.in_file, sep='\t', header=None, names=['text', 'dock_score','label']) # Pandas identifies automatically if we use int64 or float per column
        data.drop('dock_score', axis=1, inplace=True)
    
    else:

        X, y = parsers.parseAMCPDataTraining(args)

        # data = pd.read_csv(args.in_file, sep=' ', header=None) # Pandas identifies automatically if we use int64 or float per column

    for model_iteration in range(args.numberModels):

        if args.justmodel is not None and args.justmodel != model_iteration + 1: continue

        logger.info(f'Now building model: {model_iteration + 1}')

        if args.classifier=='BERT':

            X_train, X_calibration_data, y_train, y_calibration_data = \
                train_test_split(data['text'], data['label'], test_size=args.ratioTestSet, shuffle=True, stratify=data['label'])

            # Calculate number of training examples for each class (for weights)
            len_train_1 = np.count_nonzero(y_train) #len([x for x in y_train_model if x==1])
            len_train_0 = y_train.shape[0] - len_train_1 # len([x for x in y_train_model if x==0])

            logger.info(f'Number of elements in class 1: {len_train_1} ')
            logger.info(f'Number of elements in class 0: {len_train_0} ')


            ml_model = ml_models.generateModel(args, len_train_0, len_train_1)

            train_df = pd.concat([X_train, y_train], axis=1)
            calibration_df = pd.concat([X_calibration_data, y_calibration_data], axis=1)

            y_calibration_data = y_calibration_data.to_numpy()

            ml_model.train_model(train_df, eval_df=calibration_df, output_dir=os.path.join(models_dir,'run_' + str(model_iteration)))

        else:

            try:

                X_train, X_calibration_data, y_train, y_calibration_data = train_test_split(X, y, test_size=args.ratioTestSet, shuffle=True, stratify=y)

            except:

                logger.warning('Trying again the split for problems with the int32')

                X_train, X_calibration_data, y_train, y_calibration_data = train_test_split(X, y, test_size=args.ratioTestSet, shuffle=True, stratify=y)


            # Calculate number of training examples for each class (for weights)
            len_train_1 = np.count_nonzero(y_train) #len([x for x in y_train_model if x==1])
            len_train_0 = y_train.shape[0] - len_train_1 # len([x for x in y_train_model if x==0])

            logger.info(f'Number of elements in class 1: {len_train_1} ')
            logger.info(f'Number of elements in class 0: {len_train_0} ')


            ml_model = ml_models.generateModel(args, len_train_0, len_train_1)

            if args.classifier == "catboost": 

                eval_dataset = Pool(X_calibration_data, y_calibration_data)
                
                ml_model.fit(X_train, y_train, use_best_model=True, eval_set=eval_dataset)
            
            else: 

                ml_model.fit(X_train, y_train)

        # Saving models to disk.
        joblib.dump(ml_model, f'{models_dir}/amcp_clf_m{model_iteration}.z')

        if args.classifier=='BERT':

            ml_model.model.to(torch.device('cpu'))

            joblib.dump(ml_model, f'{models_dir}/amcp_clf_m{model_iteration}_cpu.z')

        # Retrieving the calibration conformity scores.

        for c, alpha_c in enumerate(amcp_lib.cali_nonconf_scores(X_calibration_data, y_calibration_data, ml_model)):
            
            joblib.dump(alpha_c, f'{models_dir}/amcp_cal_alphas{c}_m{model_iteration}.z')

        
def predict(args):
    """Reads the pickled ml_models and calibration conformity scores.
    The median p-values are calculated and written out in the out_file.

    Args:
        args (_type_):User arguments
    """

    logger.info('Starting predict')

    models_dir = args.models_dir

    os.makedirs(models_dir, exist_ok=True)
    logger.info(f'Loading classifiers from: {models_dir}')

    # if os.path.exists(args.out_file): os.remove(args.out_file)
    if os.path.exists(args.out_file + '.bz2'): os.remove(args.out_file + '.bz2')

    # Reading parameters
    smooth_bool = False if args.nosmooth else True

    dir_files = os.listdir(models_dir)
    nr_of_model_files = sum([1 for f in dir_files if f.endswith(".z")])
    nr_of_models = sum([1 for f in dir_files if f.startswith("amcp_clf")])

    # Number of classes is amount of model files, divided by model count
    #  minus 1, as 1 file is classification file
    nr_class = int((nr_of_model_files/nr_of_models)-1)

    # Initializing list of pointers to ml_model objects
    #  and calibration conformity score lists.
    ml_models = []
    calibration_alphas_c = [list() for c in range(nr_class)]

    for i in range(nr_of_models):

        try:

            ml_models.append(joblib.load(f'{models_dir}/amcp_clf_m{i}.z'))

        except RuntimeError:

            # Loading compressed models and calibration conformity scores.
            ml_model = joblib.load(f'{models_dir}/amcp_clf_m{i}_cpu.z')
            ml_model.device(torch.device('cpu'))
            ml_models.append(ml_model)
            
        for c in range(nr_class): 
            calibration_alphas_c[c].append(joblib.load(f'{models_dir}/amcp_cal_alphas{c}_m{i}.z'))
        
    logger.info(f'Loaded {nr_of_models} models.')

    modelType = type(ml_models[0]).__name__

    for idx, filein in enumerate(args.in_file):

        if modelType == 'ClassificationModel':  # BERT model
            
            data = pd.read_csv(filein, header=None, names=['text', 'dock_score','label'], chunksize=100000, delim_whitespace=True) # Pandas identifies automatically if we use int64 or float per column
            # data.drop('dock_score', axis=1, inplace=True)

        else:

            data = pd.read_csv(filein, header=None, chunksize=100000, delim_whitespace=True)

        # with open(args.out_file, 'a') as fout:

        with bz2.BZ2File(args.out_file + '.bz2', 'a') as fout:

            for idx_chunk, chunk in enumerate(data):

                if modelType == 'ClassificationModel':  # BERT model

                    sampleIDs = np.array(chunk['dock_score'])
                    predict_data = np.array(chunk['text'])
                    real_value = np.array(chunk['label'])


                else:

                    sampleIDs = np.array(chunk.iloc[:,0])
                    predict_data = np.array(chunk.iloc[:,1:], dtype=np.float32)

                nrow = predict_data.shape[0]    
                
                # Three dimensional class array
                p_c_array = np.empty((nrow, nr_of_models, nr_class))
                
                if idx == 0 and idx_chunk == 0:

                    class_string = "\t".join(['p(%d)'%c for c in range(nr_class)])
                    # c = bz2.compress()
                    fout.write(f'amcp_prediction\tpredict_file:"{args.in_file}"\nsampleID\treal_class\t{class_string}\n'.encode('utf-8'))
                    
                    # fout.write(f'amcp_prediction\tpredict_file:"{args.in_file}"\nsampleID\treal_class\t{class_string}\n')


                for model_index, ml_model in enumerate(ml_models):
                    # Predicting and getting p_values for each model
                    #  and sample.

                    if modelType == 'ClassificationModel':  # BERT model
                        
                        predict_alphas = amcp_lib.get_Nonconformity_scores(ml_model, predict_data)

                    else:

                        predict_alphas = amcp_lib.get_Nonconformity_scores(ml_model, predict_data[:, 1:])

                    # Iterate over the classes
                    for c, predict_alpha_c in enumerate(zip(*predict_alphas)):
                        for sample_index in range(nrow):

                            p_c = amcp_lib.get_CP_p_value(predict_alpha_c[sample_index], calibration_alphas_c[c][model_index], smooth_bool)
                            p_c_array[sample_index, model_index, c] = p_c
                
                    # Calculating median p for each sample in the array, class c
                    p_c_medians = np.median(p_c_array, axis=1)

                # Writing out sample prediction.
                logger.info(f'Predicted samples: {nrow}')
                for i in range(nrow):
                    p_c_string = "\t".join([str(p_c_medians[i,c]) for c in range(nr_class)])

                    if modelType == 'ClassificationModel': 

                        fout.write(f'{sampleIDs[i]}\t{real_value.iloc[i]}\t{p_c_string}\n'.encode('utf-8'))  # ???????
                    
                    else:

                        fout.write(f'{sampleIDs[i]}\t{predict_data[i, 0]}\t{p_c_string}\n'.encode('utf-8'))




def validation(args):
    """Cross validation using the K-fold method to estimate the optimal significance value.

    Args:
        args (_type_): User arguments
    """

    logger.info('Starting VALIDATION (cross-validation)')

    from sklearn.model_selection import StratifiedKFold

    # Reading parameters

    smooth_bool = False if args.nosmooth else True
    nr_of_build_models =  args.numberModels 
    prop_test_ratio = args.ratioTestSet 
    val_folds = args.validationFolds 

    logger.info(f'Test ratio: {prop_test_ratio}')
    logger.info(f'Number of models: {nr_of_build_models}')
    logger.info(f'k-folds: {val_folds}')
        
    training_IDs, training_data, nr_class, y_label = parsers.parseAMCPData(args)

    with open(args.out_file, 'w+') as fout_test:

        kf = StratifiedKFold(n_splits=val_folds) 
        
        if args.out_file_train:

            class_string = "\t".join(['p(%d)'%c for c in range(nr_class)])

            fout_train = open(args.out_file_train, 'w+')
            fout_train.write(f'amcp_validation\ttrain_samples\t validation_file:"{args.in_file}"\nsampleID\treal_class\t{class_string}\n')


        for kfold_idx, (train_index, test_index) in enumerate(kf.split(training_data, y=y_label)):

            if args.classifier=='BERT':

                X_train, X_test = training_data[train_index], training_data[test_index]
                y_train, y_test = y_label[train_index], np.array(y_label[test_index], dtype=np.int64)

            else:

                if args.sparse:

                    X_train, X_test = training_data[train_index, :], training_data[test_index, :]
                    y_train, y_test = y_label[train_index], y_label[test_index]

                else:

                    X_train, X_test = np.array(training_data[train_index, :], dtype=np.float32), np.array(training_data[test_index, :], dtype=np.float32)
                    y_train, y_test = y_label[train_index], y_label[test_index]
                    # y_train, y_test = np.array(training_data[train_index, 0], dtype=np.int64), np.array(training_data[test_index, 0], dtype=np.int64)

            nr_of_test_samples = y_test.shape[0]
            nr_of_training_samples = y_train.shape[0]

            # Three dimensional class array. Initialize/reset.
            p_c_array = np.empty((nr_of_test_samples, nr_of_build_models, nr_class), dtype=float) 

            if args.out_file_train:
                p_c_array_training = np.empty((nr_of_training_samples, nr_of_build_models, nr_class), dtype=float)
                logger.info(f'Allocated memory for array.')

            for model_iteration in range(nr_of_build_models):

                logger.info(f'Now building model: {model_iteration} for kfold: {kfold_idx}')

                X_train_model, X_calibration_data, y_train_model, y_calibration_data = train_test_split(X_train, y_train, test_size=prop_test_ratio, shuffle=True)

                # Calculate number of training examples for each class (for weights)
                len_train_1 = np.count_nonzero(y_train_model) #len([x for x in y_train_model if x==1])
                len_train_0 = y_train_model.shape[0] - len_train_1 # len([x for x in y_train_model if x==0])

                logger.info(f'Number of elements in class 1: {len_train_1} ')
                logger.info(f'Number of elements in class 0: {len_train_0} ')

                ml_model = ml_models.generateModel(args, len_train_0, len_train_1)

                if args.classifier=='BERT':
        
                    train_df = pd.concat([X_train, y_train], axis=1)
                    calibration_df = pd.concat([X_calibration_data, y_calibration_data], axis=1)

                    ml_model.train_model(train_df, eval_df=calibration_df)

                    y_calibration_data = y_calibration_data.to_numpy()
            
                else:

                    if args.classifier == "catboost": 

                        eval_dataset = Pool(X_calibration_data, y_calibration_data)
                        ml_model.fit(X_train_model, y_train_model, use_best_model=True, eval_set=eval_dataset)
                
                    else: 

                        ml_model.fit(X_train_model, y_train_model)

                calibration_alphas_c = [alpha_c for alpha_c in amcp_lib.cali_nonconf_scores(X_calibration_data, y_calibration_data, ml_model)]

                test_alphas = amcp_lib.get_Nonconformity_scores(ml_model, X_test)
                if args.out_file_train: training_alphas = amcp_lib.get_Nonconformity_scores(ml_model, X_train)
                
                # Iterate over the classes.
                for c, test_alpha_c in enumerate(zip(*test_alphas)):
                    for sample_index in range(nr_of_test_samples):
                        p_c = amcp_lib.get_CP_p_value(test_alpha_c[sample_index], calibration_alphas_c[c], smooth_bool)
                        p_c_array[sample_index, model_iteration, c] = p_c

                # Iterate over the classes for the training samples.
                if args.out_file_train:
                    for c, training_alpha_c in enumerate(zip(*training_alphas)):
                        for sample_index in range(nr_of_training_samples):
                            p_c = amcp_lib.get_CP_p_value(training_alpha_c[sample_index], calibration_alphas_c[c], smooth_bool)
                            p_c_array_training[sample_index, model_iteration, c] = p_c

            # Calculating median p for each sample in the array, class c
            p_c_medians = np.median(p_c_array, axis=1)

            # Writing out sample prediction.
            for i in range(nr_of_test_samples):
                p_c_string = "\t".join([str(p_c_medians[i,c]) for c in range(nr_class)])
                fout_test.write(f'{training_IDs[test_index[i]]}\t{y_test[i]}\t{p_c_string}\n')

            # Calculating median p for each sample in the array, class c
            if args.out_file_train:
                p_c_medians_training = np.median(p_c_array_training, axis=1)
                for i in range(nr_of_training_samples):
                    p_c_string = "\t".join([str(p_c_medians_training[i,c]) for c in range(nr_class)])
                    fout_train.write(f'{training_IDs[train_index[i]]}\t{y_train[i]}\t{p_c_string}\n')

        if args.out_file_train:
            fout_train.close()

