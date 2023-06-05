import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from scipy.sparse import csr_matrix
from scipy.sparse import vstack

import logging
logger = logging.getLogger(__name__)

def parseAMCPData(args):

    logger.info(f'Parsing data file: {args.in_file}')


    if args.classifier=='BERT':
    
        data = pd.read_csv(args.in_file, sep='\t', header=None, names=['text', 'dock_score','label']) # Pandas identifies automatically if we use int64 or float per column
        training_IDs = data['dock_score'].to_numpy()

        training_data = data['text']
        nr_class = len(np.unique(data['label']))
        y_label = data['label']

    else:

        if args.parquet: 
            
            data = pq.ParquetFile(args.in_file)
            f_to_iterate = enumerate(data.iter_batches(batch_size=1_000_000))

        else: 

            data = pd.read_csv(args.in_file, sep=' ', header=None, chunksize=100000)
            f_to_iterate = enumerate(data)

        nr_class_set = set([0])

        for idx, batch in f_to_iterate:

            chunk = batch
            if args.parquet: chunk = batch.to_pandas()

            if idx == 0: 

                training_IDs = chunk.iloc[:,0].to_numpy()
                y_label = chunk.iloc[:,1].to_numpy(dtype=np.int8)

                if args.sparse: training_data = csr_matrix(chunk.iloc[:,2:].to_numpy())
                else: training_data = chunk.iloc[:,2:].to_numpy()
            
            else: 
                
                if args.parquet:

                    training_IDs = np.hstack((training_IDs, chunk.iloc[:,0].to_numpy()))
                    y_label = np.hstack((y_label, chunk.iloc[:,1].to_numpy(dtype=np.int8)))

                else:

                    training_IDs = np.vstack((training_IDs, chunk.iloc[:,0].to_numpy()))
                    y_label = np.vstack((y_label, chunk.iloc[:,1].to_numpy(dtype=np.int8)))



                if args.sparse: training_data = vstack((training_data, csr_matrix(chunk.iloc[:,2:].to_numpy())))
                else: training_data = np.vstack((training_data, chunk.iloc[:,2:].to_numpy()))

            nr_class_set.update(np.unique(y_label.flatten()))

            logger.info(f'Processed {idx} M molecules')

            # print(training_IDs.shape)
            # print(y_label.shape)
            # print(training_data.shape)

        del data

        nr_class = len(nr_class_set)

        training_IDs = training_IDs.flatten()
        y_label = y_label.flatten()


        # columnNames = data.get_column_names()

        # training_IDs = data[columnNames[0]].to_numpy()
        # y_label = data[columnNames[1]].to_numpy().astype(np.int8)

        # if args.sparse: training_data = csr_matrix(data[columnNames[2:]])
        # else: training_data = np.array(data[columnNames[2:]])

        # nr_class = np.unique(y_label.flatten()).shape[0]

        # REGULAR

        # data = pd.read_csv(args.in_file, sep=' ', header=None)

        # data = pd.read_csv(args.in_file, sep=' ', header=None)
        # training_IDs = data.iloc[:,0].to_numpy()

        # training_data = data.iloc[:,2:].to_numpy()
        # nr_class = len(np.unique(training_data[:,0]))
        # y_label = training_data[:, 0]



        # data = pd.read_csv(args.in_file, sep=' ', header=None, chunksize=100000)

        # nr_class_set = set([0])

        # for idx, chunk in enumerate(data):

        #     if idx == 0: 

        #         training_IDs = chunk.iloc[:,0].to_numpy()
        #         y_label = chunk.iloc[:,1].to_numpy()

        #         if args.sparse: training_data = csr_matrix(chunk.iloc[:,2:].to_numpy())
        #         else: training_data = chunk.iloc[:,2:].to_numpy()
            
        #     else: 
                
        #         training_IDs = np.vstack((training_IDs, chunk.iloc[:,0].to_numpy()))
        #         y_label = np.vstack((y_label, chunk.iloc[:,1].to_numpy(dtype=np.int8)))

        #         if args.sparse: training_data = vstack((training_data, csr_matrix(chunk.iloc[:,2:].to_numpy())))
        #         else: training_data = np.vstack((training_data, chunk.iloc[:,2:].to_numpy()))

        #     nr_class_set.update(np.unique(y_label.flatten()))

        # nr_class = len(nr_class_set)

        # training_IDs = training_IDs.flatten()
        # y_label = y_label.flatten()

    # del data

    return training_IDs, training_data, nr_class, y_label


def parseAMCPDataTraining(args):

    logger.info(f'Parsing data file for training: {args.in_file}')

    if args.parquet: 
        
        data = pq.ParquetFile(args.in_file)
        f_to_iterate = enumerate(data.iter_batches(batch_size=1_000_000))

    else: 

        data = pd.read_csv(args.in_file, sep=' ', header=None, chunksize=100000)
        f_to_iterate = enumerate(data)


    for idx, batch in f_to_iterate:

        chunk = batch
        if args.parquet: chunk = batch.to_pandas()

        if idx == 0: 

            y = chunk.iloc[:,1].to_numpy()

            if args.sparse: X = csr_matrix(chunk.iloc[:,2:].to_numpy())
            else: X = chunk.iloc[:,2:].to_numpy()
        
        else: 
            
            if args.parquet:

                y = np.hstack((y, chunk.iloc[:,1].to_numpy(dtype=np.int8)))

            else:

                y = np.vstack((y, chunk.iloc[:,1].to_numpy(dtype=np.int8)))

            if args.sparse: X = vstack((X, csr_matrix(chunk.iloc[:,2:].to_numpy())))
            else: X = np.vstack((X, chunk.iloc[:,2:].to_numpy()))

        logger.info(f'Processed {idx} M molecules')

    return X, y.reshape(-1,1)