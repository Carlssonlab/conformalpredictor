
import numpy as np
import pandas as pd

from . import amcp_lib

import logging
logger = logging.getLogger(__name__)


def analysisPrediction(in_file: str, out_file: str, significanceInputLst: list, smilesFile: str) -> None:
    """Analyse predictions to extract active compounds

    Args:
        in_file (str): Input file name
        out_file (str): Ouput file name
        significanceInputLst (list): Significance values
        smilesFile (str): File containing the smiles
    """

    significance = significanceInputLst[0]

    df_actives = pd.DataFrame()

    if smilesFile != None:

        df_smiles = pd.read_csv(smilesFile, header=None, usecols=[0,1], names = ['smiles', 'molID'], delim_whitespace=True) 

    for filein in in_file:

        df = pd.read_csv(filein, names=['sampleID', 'real_class', 'p0',	'p1'], skiprows=2, chunksize=100000, delim_whitespace=True) 

        for df_chunk in df:

            df_chunk['prediction'] = df_chunk.apply(lambda x: amcp_lib.set_prediction(x.p0, x.p1, significance), axis = 1)
            df_chunk['deltaP'] = df_chunk.apply(lambda x: np.abs(x.p0-x.p1), axis=1)

            df_chunk_actives = df_chunk[df_chunk['prediction'] == '{1}']

            if smilesFile != None:

                df_chunk_actives = df_chunk_actives.merge(df_smiles, left_on='sampleID', right_on='molID', how='left')
                df_chunk_actives.drop(['molID', 'real_class', 'p0', 'p1', 'prediction'], axis = 1, inplace=True)

            df_actives = pd.concat([df_actives, df_chunk_actives])


    df_actives.sort_values(by=['deltaP'], inplace=True, ascending=False)

    if smilesFile != None:

        df_actives.to_csv(out_file, columns=['smiles','sampleID','deltaP'], index=False, sep='\t')

    else:

        df_actives.to_csv(out_file, columns=['sampleID','deltaP'], index=False, sep='\t')


def generateSummary(in_file: str, out_file: str, significanceLst: list) -> np.array:
    """Generate summary function to add confusion data columns (true positives, ...)

    Args:
        in_file (str): Input file name
        out_file (str): Output file name
        significanceLst (list): Significance values to explore

    Returns:
        np.array: New summar columns
    """

    summary_array = np.zeros((len(significanceLst), 8), dtype=int)
        
    with open(in_file, 'r') as fin:

        next(fin)
        next(fin)

        for line in fin:

            line_list = line.strip().split()
            real_class = line_list[1]
            p0 = float(line_list[2])
            p1 = float(line_list[3])

            set_preds = [ amcp_lib.set_prediction(p0, p1, significance) for significance in significanceLst ]

            # Confusion matrix and set counts.

            for i, set_pred in enumerate(set_preds):

                if set_pred == '{0}':
                
                    if real_class == '0' or real_class == '0.0':
                        summary_array[i, 2] += 1 # true_neg
                    elif real_class == '1' or real_class == '1.0':
                        summary_array[i, 3] += 1 # false_neg
                
                elif set_pred == '{0,1}':
                    if real_class == '0' or real_class == '0.0':
                        summary_array[i, 4] += 1 # both_class0
                    elif real_class == '1' or real_class == '1.0':
                        summary_array[i, 5] += 1 # both_class1
                
                elif set_pred == '{}':
                    if real_class == '0' or real_class == '0.0':
                        summary_array[i, 6] += 1 # null_class0
                    elif real_class == '1' or real_class == '1.0':
                        summary_array[i, 7] += 1 # null_class1
                
                elif set_pred == '{1}':
                    if real_class == '1' or real_class == '1.0':
                        summary_array[i, 0] += 1 # true_pos
                    elif real_class == '0' or real_class == '0.0':
                        summary_array[i, 1] += 1 # false_pos

    return summary_array

def generateAnalysis(out_file: str, summary_array: np.array, significance_list: list):
    """ Calculates standard ML metrics and CP metrics based on the set/class-counts. Writes a summary file.

    Args:
        out_file (str): Output file name
        summary_array (np.array): Summary data to analyse
        significance_list (list): Significances to check
    """

    with open(out_file, 'w+') as fout:

        fout.write('significances\ttrue_pos\tfalse_pos\ttrue_neg\tfalse_neg\t'
                'both_class0\tboth_class1\tnull_class0\tnull_class1\t'
                'error_rate\terror_rate_class1\terror_rate_class0\t'
                'efficiency\tefficiency_class1\tefficiency_class0\t'
                'precision\tsensitivity\tf1_score\tpred1_ratio'
                '\tharmonic\n')

        total_samples = sum(summary_array[0, :])
        total_positives = (summary_array[0, 0] + summary_array[0, 7] + summary_array[0, 5] + summary_array[0, 3])
        total_negatives = (summary_array[0, 2] + summary_array[0, 4] + summary_array[0, 6] + summary_array[0, 1])

        for i, significance in enumerate(significance_list):

            precision = 'n/a'
            sensitivity = 'n/a'
            f1_score = 'n/a'
            pred1_ratio = 'n/a'
            harmonic = 'n/a'

            # (false_pos + false_neg + null_set) / total_samples
            error_rate = (summary_array[i, 1] + summary_array[i, 3] + summary_array[i, 6] + summary_array[i, 7]) / total_samples

            # (true_pos + both_class1) / total_positives.
            error_rate_class1 = (summary_array[i, 3] + summary_array[i, 7]) / total_positives

            # (true_neg + both_class0) / total_negatives.
            error_rate_class0 = (summary_array[i, 6] + summary_array[i, 1]) / total_negatives

            # (true_pos + false_pos + true_neg + false_neg) / total_samples
            efficiency = (summary_array[i, 0] + summary_array[i, 1] + summary_array[i, 2] + summary_array[i, 3]) / total_samples

            # (true_pos + false_neg) / total_samples
            efficiency_1 = (summary_array[i, 0] + summary_array[i, 3]) / total_positives

            # (true_pos g) / total_samples
            efficiency_0 = (summary_array[i, 1] + summary_array[i, 2]) / total_negatives

            if summary_array[i, 0] + summary_array[i, 1] != 0:
                # true_pos / (true_pos + false_pos)
                precision = summary_array[i, 0] / (summary_array[i, 0] + summary_array[i, 1])

            if total_positives != 0:
                # Normally:
                #  true_pos / (true_pos + false_neg), AKA recall.
                # In CP:
                #  true_pos / total_positives.
                sensitivity = summary_array[i, 0] / total_positives

            if precision != 'n/a' and sensitivity != 'n/a':
                f1_score =  (2 * precision * sensitivity) / (precision + sensitivity)

            if total_samples != 0 and total_samples != 'n/a':
                # (true_pos + false_pos) / total_samples
                pred1_ratio = (summary_array[i, 0] + summary_array[i, 1]) / total_samples

            if precision != 'n/a' and sensitivity != 'n/a' and pred1_ratio != 'n/a':
                # Harmonic mean for sensitivity, precision and 1-pred1_ratio.
                harmonic = (3*(1-pred1_ratio)*precision*sensitivity) / ((1-pred1_ratio)*precision + (1-pred1_ratio)*sensitivity+ (precision*sensitivity))

            if f'{significance:.3}' == '1.0': continue

            fout.write(f'{significance:.3}\t{summary_array[i,0]}\t'
                    f'{summary_array[i,1]}\t{summary_array[i,2]}\t'
                    f'{summary_array[i,3]}\t{summary_array[i,4]}\t'
                    f'{summary_array[i,5]}\t{summary_array[i,6]}\t'
                    f'{summary_array[i,7]}\t{error_rate:.3}\t'
                    f'{error_rate_class1:.3}\t{error_rate_class0:.3}\t'
                    f'{efficiency:.3}\t{efficiency_1:.3}\t{efficiency_0:.3}\t'
                    f'{precision:.3}\t{sensitivity:.3}\t'
                    f'{f1_score:.3}\t{pred1_ratio:.3}\t{harmonic:.3}\n')



def analysis(in_file: str, out_file: str, significanceInputLst: list) -> None:
    """Analysis of the AMCP prediction run

    Args:
        in_file (str): Input file
        out_file (str): Output file
        significance (list): Significance list
    """

    logger.info('Starting ANALYSIS of the prediction run')

    significanceLst = significanceInputLst

    if significanceLst == None: significanceLst = list(np.linspace(0,1.0,num=50))

    summary_array = generateSummary(in_file, out_file, significanceLst)

    generateAnalysis(out_file, summary_array, significanceLst)

