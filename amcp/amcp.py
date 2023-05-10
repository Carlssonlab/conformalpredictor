import argparse
import os
import sys

from time import perf_counter
from pathlib import Path

from . import modes
from . import plots
from . import analysis

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename="amcp.log", filemode='a', level=logging.DEBUG,format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S',)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.ERROR)

logger.addHandler(stream_handler)

def argument_parser(arguments) -> argparse:
    """Argument parser

    Returns:
        argparse: Arguments
    """
    class UltimateHelpFormatter(
        argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    def restricted_float(x):
        """Restrict values from 0.0 to 1.0

        Args:
            x (float): User argument
        Returns:
            float: User argument
        """

        try:
        
            x = float(x)
        
        except ValueError:
        
            raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")

        if x < 0.0 or x > 1.0:

            raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 1.0]")
        
        return x



    info = """Aggregated Mondrian Conformal Predictor (AMCP).

    Ex. input file (space separated file):

    
        CP001892232805_-27.90 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 ... 0 0
        CP002491258647_-28.51 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ... 0 0
        CP001136835595_-34.41 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ... 0 0
        CP002361737404_-34.11 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ... 0 0
        CP000831597329_-32.93 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ... 0 1

    Ex. run

    amcp -m validation -i validation_features.txt -o validation_result_output.txt
    amcp -m train -i input_train_amcp.txt -o train_out.txt
    amcp -m prediction -i test_features.txt -o test_prediction_output.txtt

    """


    parser = argparse.ArgumentParser(description='AMCP', epilog=info, formatter_class=UltimateHelpFormatter)

    parser.add_argument('-cf', '--classifier', choices=[ 'catboost', 'dnn', 'BERT'], default='catboost', help='Choose the underlying classifier')
    parser.add_argument('-i', '--in_file', nargs='+', help='(for \'build, \'validation\', \'predict\') Specify the file containing the data.', required=True)
    parser.add_argument('-md', '--models_dir', help='(for \'build\', \'predict\') Specify the absolute directory where the generated models should be (for build) or are (for \'predict\') placed. Otherwise it will be the directory amcp_models.', type=str, default=f'{os.getcwd()}/amcp_models/')
    parser.add_argument('-o', '--out_file', help='(for \'predict\', \'validation\') Specify the output file.')
    parser.add_argument('-m', '--mode', choices=['train', 'validation', 'predict', 'analysis'], help='Choose script mode.', required=True)
    parser.add_argument('-nm', '--numberModels', help='Number of models to use for the ensemble', default=5, type=int)
    parser.add_argument('-rt', '--ratioTestSet', help='Test set size ratio', default=0.2, type=restricted_float)
    parser.add_argument('-sig', '--significance', help='Significance to use in prediction analysis', nargs='+', default=None, type=restricted_float)
    parser.add_argument('-vf', '--validationFolds', help='Folds for cross validation', default=5, type=int)
    parser.add_argument('-s', '--nosmooth', help='Not use smooth', action='store_true')
    parser.add_argument('-dlr','--dnnLearningRate', help="DNN learning rate", default=1e-4, type=restricted_float)
    parser.add_argument('-blr','--BERTLearningRate', help="BERT learning rate", default=4e-7, type=restricted_float)
    parser.add_argument('-dwd','--dnnWeightDecay', help="DNN weight decay", default=1e-2, type=restricted_float)
    parser.add_argument('-dme','--dnnMaxEpoch' ,help="DNN max epochs", default=100, type=int)
    parser.add_argument('-bme','--BERTMaxEpoch', help="BERT max epochs", default=10, type=int)
    parser.add_argument('-parquet', '--parquet', help='Not use smooth', action='store_true')
    parser.add_argument('-sparse', '--sparse', help='Use sparse matrices to save memory', action='store_true')
    parser.add_argument('-dnn', '--dnnLayers', help='DNN layers', default=[1024, 1000, 4000, 2000], type=list, nargs='+')
    parser.add_argument('-fpsize', '--fpsize', help='Size of the fingerPrint bit representation', default=1024, type=int)
    parser.add_argument('-justmodel', '--justmodel', help='Only train the model number X. For parallelization of training. Starts at 1', default=None, type=int)
    parser.add_argument('-oft', '--out_file_train', help='(for \'validate\') Predict the training samples.', default=None)
    parser.add_argument('-smi', '--smiles', help='Smiles file to match data', default=None, type=str)

    args = parser.parse_args(arguments)

    for inFile in args.in_file:
        if not Path(inFile).is_file():
            parser.error(f'The input file: {inFile} does not exist.')

    return args


def main():
    
    t1_start = perf_counter()

    args = argument_parser(sys.argv[1:])

    for inFile in args.in_file:
        logger.info(f'Input file: {inFile}')

    if args.mode == 'validation' or args.mode == 'train':

        if len(args.in_file) > 1:

            logger.error(f'More than one inputFile for  {args.mode} NOT allowed!!')
            exit()
    
        args.in_file = args.in_file[0]

    if args.mode == 'validation':

        if not args.out_file: args.out_file = os.path.splitext(args.in_file)[0] + '_validation.txt'
        if not args.out_file_train: args.out_file_train = os.path.splitext(args.in_file)[0] + '_train_validation.txt'

        logger.info(f'Output file: {args.out_file}')
        if args.out_file_train != None and args.mode == 'validation': logger.info(f'Validation output train file: {args.out_file_train}')

        modes.validation(args)

        summaryFileNames = os.path.splitext(args.out_file)[0] + '_summary.txt'

        analysis.analysis(args.out_file, summaryFileNames, args.significance)

        plots.plotValidationResults(summaryFileNames) 
    
    elif args.mode == 'train':

        if not args.out_file: args.out_file = os.path.splitext(args.in_file)[0] + '_train.txt'
        logger.info(f'Output file: {args.out_file}')

        modes.train(args)
    
    elif args.mode == 'predict':

        if not args.out_file: args.out_file = os.path.splitext(args.in_file[0])[0] + '_prediction.txt'
        logger.info(f'Output file: {args.out_file}')

        modes.predict(args)

    elif args.mode == 'analysis':

        if not args.significance: 

            logger.error(f'Significance value required ( -sig ). Check amcp.log for the optimal significance value extracted from the validation')
            exit()

        if not args.out_file: args.out_file = os.path.splitext(args.in_file[0])[0] + '_topActives.txt'
        logger.info(f'Output file: {args.out_file}')

        analysis.analysisPrediction(args.in_file, args.out_file, args.significance, args.smiles)


    logger.info(f'Running time: {(perf_counter() - t1_start):.2f} s.')


if __name__ == '__main__':
    
    main()
