__author__  = 'Israel Cabeza de Vaca Lopez'
__date__    = '2023-03-05'
__email__   = 'israel.cabezadevaca@icm.uu.se'
__version__ = '1.0'

info = """Plot distributions for the outputs from the Agreegated Mondrian Conformal Predictor (AMCP) training and prediction docking results.
Ex. run
amcp_analyseScoreDistributions -train extract_all.sort.uniq_train.txt -pred extract_all.sort.uniq_pred.txt -o scoreDistributions.txt    (training)

"""

import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from time import perf_counter

class UltimateHelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass

def argument_parser():
    """Argument parser
    Returns:
        argparser: Arguments parsed
    """

    parser = argparse.ArgumentParser(description='AMCP score distribution analysis from DOCK', epilog=info, formatter_class=UltimateHelpFormatter)
    parser.add_argument('-train', '--train_file', help='DOCK3.7 results file for training data.', type=str, required=True)
    parser.add_argument('-pred', '--pred_file', help='DOCK3.7 results file for AMCP predicted set.', type=str, required=True)
    parser.add_argument('-o', '--out_file', help='Name of the output plot file (PNG)', type=str, required=True)

    args = parser.parse_args()

    return args


def main():
    
    args = argument_parser()
    
    cols = pd.read_csv(args.train_file, nrows=1, delim_whitespace=True).columns
    
    df_train = pd.read_csv(args.train_file, delim_whitespace=True, usecols=[len(cols)-1], header=None, names=['scores'])
    df_production = pd.read_csv(args.pred_file, delim_whitespace=True, usecols=[len(cols)-1], header=None, names=['scores'])

    # Plot the distributions
    sns.distplot(df_production, hist=False, kde=True, label='Predicted')
    sns.distplot(df_train, hist=False, kde=True, label='Training')

    # Add axis labels and title
    plt.xlabel('Dock3.7 scores', fontsize=16, fontweight='bold')
    plt.ylabel('Density', fontsize=16, fontweight='bold')
    plt.title('PLpro AMCP Score distributions')
    plt.legend(['Predicted', 'Training'])

    plt.savefig(args.out_file, dpi=300)

if __name__ == '__main__':
    
    main()