__author__  = 'Israel Cabeza de Vaca Lopez'
__date__    = '2022-10-15'
__email__   = 'israel.cabezadevaca@icm.uu.se'
__version__ = '1.0'

info = """Preparation of the inputs for Agreegated Mondrian Conformal Proedictor (AMCP).

Ex. input file (space separated file):

Cc1ccc(cc1Cl)C=CC(=O)N2CCC(C2)(C)CNc3c(nccn3)C#N CP000469650145 -63.64
Cc1c(cc(cn1)NC(=O)C)C(=O)N2CCN(CC23CCC3)C(=O)C=CC(=O)N CP000105192859 -59.13
CC(C1CN(CCO1)C(=O)c2cc3cc[nH]c(=O)c3nc2)NC(=O)c4[nH]ccn4 CP000261423010 -58.56
CC1CN(CCN1C(=O)c2cc(cnc2)N)C(=O)c3ccc(cc3)OC4CCC4 CP000121223014 -58.39

Ex. run

amcp_preparation -i input_train.txt -o output_train.txt    (training)
amcp_preparation -i input_train.txt -o output_train.txt  -cpus 4  (training using 4 cpus/cores)
amcp_preparation -i input_prediction.txt -o output_prediction.txt -pred   (prediction)
amcp_preparation -i input_prediction.txt -o output_prediction.txt -pred  -c 10    (prediction - splitting outputfile in 10 files for parallel prediction)
amcp_preparation -i input_prediction.txt -o output_prediction.txt -pred  -bert    (prediction - For BERT algorithm - NOT recommende because is extreemly slow)

For a 1 M molecules training file takes around 10 min

"""

import argparse

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import os
from functools import partial

from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from time import perf_counter
import bz2

class UltimateHelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename="amcp_preparation.log", filemode='a', level=logging.DEBUG,format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S',)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.ERROR)

logger.addHandler(stream_handler)


def argument_parser():
    """Argument parser

    Returns:
        argparser: Arguments parsed
    """

    parser = argparse.ArgumentParser(description='AMCP input preparation', epilog=info, formatter_class=UltimateHelpFormatter)
    parser.add_argument('-i', '--in_file', help='Input file.', required=True)
    parser.add_argument('-o', '--out_file', help='Output file.', required=True)
    parser.add_argument('-pred', '--prediction', help='File prepared for prediction (not training).', action='store_true')
    parser.add_argument('-bert', '--bert', help='Prepare file for BERT training/prediction', action='store_true')
    parser.add_argument('-r', '--radius', help='Fingerprint radius.', type=int, default = 2)
    parser.add_argument('-s', '--sizeBits', help='Fingerprint size in bits.', type=int, default = 1024)
    parser.add_argument('-p', '--topActives', help='Top percentage of scoring function to define active class (1). Default 1 %% of the top scored', type=float, default = 1.0)
    parser.add_argument('-t', '--scorethreadhold', help='Manual docking score threadhold', type=float, default = None)
    parser.add_argument('-c', '--chunks', help='Number of chunks to split the output file (for parallel computing)', type=int, default = 1)
    parser.add_argument('-cpus', '--cpus', help='Number of cpus (for parallel computing - default are all) ', type=int, default = None)
    parser.add_argument('-nocompress', '--nocompress', help='Do NOT compress outputs', action='store_true')

    args = parser.parse_args()

    return args

def findThresHold(in_file: str, topActives: float) -> float:
    """Find the top percentage value to use as a threadhold

    Args:
        in_file (str): Input file
        topActives (float): Top scores considered actives

    Returns:
        float: Score corresponding to the active limit
    """

    logger.info(f'Finding threshold ')

    df = pd.read_csv(in_file, header=None, delim_whitespace=True)

    if len(df.columns) != 3:

        logger.error(f'Error in input file for TRAINING. Number of columns is {len(df.columns)} but should be 3.')
        exit()

    scores = df.iloc[:,-1].astype(float).sort_values()

    top_idx = int(scores.shape[0]*(topActives/100.0))

    threshold = scores.iloc[top_idx]

    return threshold

def fingerPrint(smiles: str, radius: float, sizeBits: int):
    """Generate finger print

    Args:
        smiles (str): SMILES of the molecule
        radius (float): Circular fingerprint radius
        sizeBits (int): Size of the bit representation

    Returns:
        _type_: Fingerprint
    """

    fp = np.zeros((1,), dtype=str)

    try:

        DataStructs.ConvertToNumpyArray(Chem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius, nBits = sizeBits), fp)

    except Exception as msg:

        fp = f'ERROR: {msg}'
    
    return fp

def checkSMILES(smiles: str):
    """Check SMILES for errors

    Args:
        smiles (str): SMILES of the molecule

    Returns:
        str: GOOD if SMILES is right
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None: return f'ERROR: SMILES not accepted {smiles}'

    return 'GOOD'


def getChunkSize(in_file: str, chunks: int) -> int:
    """Determine the number of lines per file according to the number of files required (chunks)

    Args:
        in_file (str): Input file name
        chunks (int): Number of output files required

    Returns:
        int: Number of lines per file
    """

    linesPerChunk = sum(1 for line in open(in_file))

    shift = 0
    if linesPerChunk % chunks != 0: shift = 1 
    
    linesPerChunk = int(float(linesPerChunk)/chunks) + shift
    
    logger.info(f'Number of lines per output file: {linesPerChunk}')

    return linesPerChunk

def generateFingerPrint(df_chunk: pd.DataFrame, radius: float, sizeBits: int, bert: bool) -> pd.DataFrame:
    """Compute molecular finger print

    Args:
        df_chunk (pd.DataFrame): Input SMILES data
        radius (float): Circular fingerprint radii value
        sizeBits (int): Fingerprint in bits size
        bert (bool): True if output is for a BERT training/prediction

    Returns:
        pd.DataFrame: Fingerprint
    """

    if bert: df_chunk['fp'] = df_chunk['smiles'].apply(lambda x: checkSMILES(x))
    else: df_chunk['fp'] = df_chunk['smiles'].apply(lambda x: fingerPrint(x, radius, sizeBits))

    df_chunk_failed = df_chunk[df_chunk.fp.str.contains('ERROR', na=False)]
    if df_chunk_failed.shape[0] != 0: df_chunk_failed.to_csv('failed_molecules.smi', mode='a', sep=' ', header=False, index=False)

    return df_chunk

def addColumns(df_chunk: pd.DataFrame, in_file: str, prediction: bool) -> pd.DataFrame:
    """Add columns names to the dataframe depending on the input

    Args:
        df_chunk (pd.DataFrame): Chunk of data of the input file
        in_file (str): Input file name
        prediction (bool): If True, input file is a prediction file

    Returns:
        pd.DataFrame: Chunk of the input data
    """

    if len(df_chunk.columns) == 3: 
        
        df_chunk.columns = ['smiles', 'molID','scores']
    
    elif len(df_chunk.columns) == 2 and prediction:  

        df_chunk.columns = ['smiles', 'molID']
        df_chunk['scores'] = 0.0
    
    elif len(df_chunk.columns) == 1 and prediction:  
        
        df_chunk.columns = ['smiles']
        df_chunk['molID'] = 'mol_' + df_chunk.index.astype(str)
        df_chunk['scores'] = 0.0

    else:

        logger.error(f'ERROR in input file:{in_file}')
        exit()

    return df_chunk

def processChunk(args: argparse,thredsHold_score_actives: float, linesPerChunk: int, df_chunk: pd.DataFrame):
    """Process each chunk to generate data

    Args:
        args (argparse): Arguments
        thredsHold_score_actives (float): Threshold value
        linesPerChunk (int): Lines per chunk
        df_chunk (pd.DataFrame): Chunk of data
    """

    idx = df_chunk.iloc[0,:].name

    logger.info(f'process chunk: {idx}')

    df_chunk = addColumns(df_chunk, args.in_file, args.prediction)

    if len(df_chunk.columns) == 3: df_chunk['class'] =  np.where(df_chunk['scores'] < thredsHold_score_actives , 1, 0)
    else: df_chunk['class'] = 0

    df_chunk = df_chunk.reset_index()

    df_chunk = generateFingerPrint(df_chunk, args.radius, args.sizeBits, args.bert)

    if not args.prediction:
        
        df_chunk['molID'] = df_chunk['molID'].astype(str)
        df_chunk['molID'] = df_chunk.apply(lambda x: x.molID+'_'+str(x.scores), axis=1)
        df_chunk['class'] = np.where(df_chunk['scores'] < thredsHold_score_actives , 1, 0)

    df_final = pd.concat([df_chunk['molID'], df_chunk['class'], df_chunk['smiles'], df_chunk['fp']] , axis=1) # pd.DataFrame(df_chunk['fp'].to_list())

    df_final = df_final[df_final['fp'].str.contains('ERROR', na=False) == False].reset_index()

    if args.bert:

        df_final = pd.concat([df_final['molID'], df_final['smiles'], df_final['class']] , axis=1)

    else:

        df_final = pd.concat([df_final['molID'], df_final['class'], pd.DataFrame(df_final['fp'].to_list())] , axis=1)

        df_final.to_csv(args.out_file, sep=' ', header=False, index=False, mode='a')
    
    del df_final


def generateOutputFile(args: argparse, thredsHold_score_actives: float)-> None:
    """Generate the output file

    Args:
        args (argparse): Arguments
        thredsHold_score_actives (float): Docking score to define the limit between active/inactive (top 1%)
    """

    linesPerChunk = 1_000

    df = pd.read_csv(args.in_file, header=None, chunksize=linesPerChunk, delim_whitespace=True)

    if os.path.exists(args.out_file): os.remove(args.out_file)

    if args.cpus == 1:

        for df_chunk in df:

            processChunk(args, thredsHold_score_actives, linesPerChunk, df_chunk)

    else:

        processChunkPartial = partial(processChunk, args, thredsHold_score_actives, linesPerChunk)

        with ProcessPoolExecutor(max_workers=args.cpus) as executor:

            results = executor.map(processChunkPartial, df)

def parallel_df_chunk(out_file: str, extension: str, compression: str, linesperchunk: int, df_chunk: pd.DataFrame) -> None:
    """Generate chunk for splitting file

    Args:
        out_file (str): Outfile name
        extension (str): Compress or not
        compression (str): Type of compression
        linesperchunk (int): Lines per chunk
        df_chunk (pd.DataFrame): Chunk of data
    """

    idx = int(df_chunk.iloc[0,:].name/ linesperchunk)

    df_chunk.to_csv(f'{os.path.splitext(out_file)[0]}_{idx}{extension}', sep=' ', header=False, mode='w', index=False, compression=compression)


def generateMultipleOutputFiles(args: argparse) -> None:
    """Generate multiple files from one output file (split file in chunks)

    Args:
        args (argparse): User arguments 
    """

    linesPerChunk = getChunkSize(args.out_file, args.chunks)

    df = pd.read_csv(args.out_file, header=None, chunksize=linesPerChunk, delim_whitespace=True)

    extension = '.bz2'
    compression = 'bz2'

    if args.nocompress: 
        extension = '.txt'
        compression = 'None'

    processChunkPartial = partial(parallel_df_chunk, args.out_file, extension, compression, linesPerChunk)
    
    with ProcessPoolExecutor(max_workers=args.cpus) as executor:

        results = executor.map(processChunkPartial, df)

def main():

    t1_start = perf_counter()

    args = argument_parser()

    thredsHold_score_actives = args.scorethreadhold

    if thredsHold_score_actives == None:
        if not args.prediction: 
            thredsHold_score_actives = findThresHold(args.in_file, args.topActives)

    if args.scorethreadhold != None or args.prediction==False: logger.info(f'Threadhold for this dataset: {thredsHold_score_actives}')

    generateOutputFile(args, thredsHold_score_actives)

    if args.chunks>1 and args.prediction: 
        
        generateMultipleOutputFiles(args)
        os.remove(args.out_file)
    
    else:

        if not args.nocompress: # and args.prediction:

            with open(args.out_file, 'rb') as fin, bz2.BZ2File(args.out_file + '.bz2', 'w') as fout:

                while True:

                    block = fin.read(1024000)
                    if not block:
                        break

                    fout.write(block)

            os.remove(args.out_file)

    logger.info(f'Running time: {(perf_counter() - t1_start):.2f} s.')

if __name__ == '__main__':

    main()
