__author__  = 'Israel Cabeza de Vaca Lopez'
__date__    = '2022-10-15'
__email__   = 'israel.cabezadevaca@icm.uu.se'
__version__ = '1.0'

info = """Generates inputs for Agreegated Mondrian Conformal Proedictor (AMCP) preparation from DOCK software outputs.

Ex. run

amcp_inputFromDock -i smiles.ism -dock extract_all.sort.uniq.txt -o output_train.txt    (training)
amcp_inputFromDock -i smiles.ism -d extract_all.sort.uniq.txt -o output_train.txt -n 1000000  (prediction)

For a 1 M molecules training file takes around 10 min

"""

import argparse

import pandas as pd

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

    parser = argparse.ArgumentParser(description='AMCP input preparation from DOCK', epilog=info, formatter_class=UltimateHelpFormatter)
    parser.add_argument('-ismi', '--in_file', help='Input SMILES file using a two columns format (SMILES, molID). ', type=str, required=True)
    parser.add_argument('-dock', '--dock_file', help='DOCK score files from a docking calculation.', type=str, required=True)
    parser.add_argument('-o', '--out_file', help='Name of the output file', type=str, required=True)
    parser.add_argument('-n', '--numberOfMolecules', help='Number of molecules to randomly extract from dock_file', type=int, default=1000_000)
    parser.add_argument('-seed', '--seed', help='Seed used for the random selection (For reproducibility)', type=int, default=1)


    args = parser.parse_args()

    return args


def main():

    t1_start = perf_counter()

    args = argument_parser()

    cols = pd.read_csv(args.dock_file, nrows=1, delim_whitespace=True).columns

    df_dock = pd.read_csv(args.dock_file, header=None, usecols=[2,len(cols)-1], names=['molId', 'dockScore'], delim_whitespace=True)

    if df_dock.shape[0] < args.numberOfMolecules:

        print(f'Warning: Number of docked molecules: {df_dock.shape[0]} Molecules to extract {args.numberOfMolecules} ---> New number of molecules to extract: {df_dock.shape[0]}')
        args.numberOfMolecules = df_dock.shape[0]

    df_smiles = pd.read_csv(args.in_file, header=None, names=['smiles', 'molId'], delim_whitespace=True)

    df_dock_subset = df_dock.sample(n=args.numberOfMolecules, axis=0, ignore_index=True, random_state=args.seed)

    final_df = pd.merge(df_dock_subset, df_smiles, on=['molId'], how='left')

    final_df_missing = final_df[final_df['smiles'].isna()]

    if final_df_missing.shape[0] > 0: 

        print(f"{final_df_missing.shape[0]} Missing smiles!!!: Check file: missingSmiles.txt")

        final_df_missing.to_csv('missingSmiles.txt', index=False, header=False, sep=' ', columns=['molId', 'dockScore']) 

        final_df.dropna(inplace=True)

    final_df.to_csv(args.out_file, index=False, header=False, sep=' ', columns=['smiles', 'molId', 'dockScore']) 


    print(f'Running time: {(perf_counter() - t1_start):.2f} s.')

if __name__ == '__main__':

    main()
