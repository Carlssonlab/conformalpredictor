import argparse
from torch import nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)

class DNN(nn.Module):

    def __init__(self, input_dim: int = 1024, hidden_dim: list = [1000, 4000, 2000], output_dim: int = 2, dropout: float = 0.2):

        super(DNN, self).__init__()

        self.numberHiddenLayers = len(hidden_dim)

        layers = [nn.Linear(input_dim, hidden_dim[0]), nn.LeakyReLU(), nn.Dropout(dropout)]

        for i in range(1, self.numberHiddenLayers):

            layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim[-1], output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):

        x = F.softmax(self.model(x), dim=-1)

        return x


def generateModel(args: argparse, lenTrainClass0: int, lenTrainClass1: int):
    """Generate ML model with optimal hyperparameters

    Args:
        args (argparse): User arguments 
        lenTrainClass0 (int): Size of the class0
        lenTrainClass1 (int): Size of the class1

    Returns:
        _type_: ML model
    """

    logger.info(f'Generating ML model: {args.classifier}')

    if args.classifier == "catboost":

        from catboost import CatBoostClassifier

        ml_model = CatBoostClassifier(auto_class_weights="Balanced", eval_metric='AUC', silent=True) # eval_metric='MCC' BalancedAccuracy Precision Recall PRAUC

    elif args.classifier == "dnn":

        from skorch import NeuralNetClassifier
        from skorch.callbacks import EarlyStopping, Checkpoint
        from torch import nn, FloatTensor

        from .optimRangerLars import RangerLars

        early_stop_patience = 10
        early_stop_threshold = 0.001
        batch_size = 200
        max_epochs = args.DNNMaxEpochs
        
        cp = Checkpoint()

        # Setup for class weights
        class_weights = 1 / FloatTensor([lenTrainClass0, lenTrainClass1])

        # Define the skorch classifier

        early_stop = EarlyStopping(patience=early_stop_patience, threshold=early_stop_threshold, threshold_mode='rel', lower_is_better=True)

        dnn_model = DNN(input_dim=int(args.dnnLayers[0]), hidden_dim= args.dnnLayers[1:-1], output_dim=int(args.dnnLayers[-1]))

        ml_model = NeuralNetClassifier(dnn_model, batch_size=batch_size, max_epochs=max_epochs,
            optimizer=RangerLars, 
            optimizer__lr=args.dnnLearningRate,
            optimizer__weight_decay=args.dnnWeightDecay,
            criterion=nn.CrossEntropyLoss,
            callbacks=[early_stop, cp],
            criterion__weight=class_weights,
            iterator_train__shuffle=True)

    elif args.classifier == "BERT":

        from simpletransformers.classification import ClassificationModel

        max_epochs = args.BERTMaxEpochs

        ml_model = ClassificationModel('roberta', 'seyonec/PubChem10M_SMILES_BPE_396_250', args={'evaluate_each_epoch': True, 
            'evaluate_during_training_verbose': True, 'no_save': True, 'num_train_epochs': max_epochs, 'auto_weights': True,
            'overwrite_output_dir': True, 'learning_rate': args.BERTLearningRate, 'save_model_every_epoch': False})
        
    return ml_model


