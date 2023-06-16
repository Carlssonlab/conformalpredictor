import matplotlib.pyplot as pt
import copy
import os
import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

def plotValidationResults(inputfile: str) -> None:
    """Plot validation data (significance Vs RelativeSetDistribution / significance  Vs ErrorRate)

    Args:
        inputfile (str): Input summary file
    """

    df_summary = pd.read_csv(inputfile, delim_whitespace=True)

    plotSignificancesVsRelativeSetDistribution(inputfile, df_summary, latex=False)

    plotSignificancesVsErrorRate(inputfile, df_summary, latex=False)

    try:

        plotSignificancesVsRelativeSetDistribution(inputfile, df_summary, latex=True)

        plotSignificancesVsErrorRate(inputfile, df_summary, latex=True)

    except RuntimeError as e:

        logger.warning(e)
        logger.warning('Plots will use a basic (no symbols) style')


def plotSignificancesVsErrorRate(inputfile: str, df: pd.DataFrame, latex: bool) -> None:
    """Generate significance VS ErrorRate plot

    Args:
        inputfile (str): Input file to plot
        df (pd.DataFrame): Data to plot
    """

    outputfile = os.path.splitext(inputfile)[0] + '_significancesVsErrorRate.png'

    significances = df['significances'].astype(float).to_list()
    errorRates = df['error_rate'].astype(float).to_list()

    pt.clf()
    pt.style.use('seaborn-white')


    if latex: pt.rc("text", usetex=True)

    fig, ax = pt.subplots(figsize=(10,10))

    pt.ylabel("Error rate", fontweight='bold',fontsize=46)

    pt.xlabel(r'significance (Epsilon)', fontweight='bold',fontsize=46)
    if latex: pt.xlabel(r'significance ($\varepsilon$)', fontweight='bold',fontsize=46)

    pt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],fontsize=36)
    pt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],fontsize=36)
    pt.tick_params(length=5, width=2.5)
    
    for axis in ['top', 'bottom', 'left', 'right']: ax.spines[axis].set_linewidth(2.5)
    
    pt.yticks(fontsize=36)
    pt.xticks(fontsize=36)
    pt.tick_params(length=10, width=3)

    pt.plot(significances, errorRates, 'bo-')

    ax.plot([0, 1], [0, 1], color='r', ls='--', linewidth=2, transform=ax.transAxes)

    fig.savefig(outputfile, bbox_inches='tight', dpi=300, transparent=False)


def plotSignificancesVsRelativeSetDistribution(inputfile: str, df: pd.DataFrame, latex: bool) -> None:
    """Generate significance VS relative set distribution plot

    Args:
        inputfile (str): Input file to plot
        df (pd.DataFrame): Data to plot
    """

    outputfile = os.path.splitext(inputfile)[0] + '_significanceVsRelativeSetDistribution.png'

    logger.info(f'Plotting significances: {inputfile}')

    significances = df['significances'].astype(float).to_list()
    ones = df['true_pos'].astype(int) + df['false_pos'].astype(int)
    zeros = df['true_neg'].astype(int) + df['false_neg'].astype(int)
    boths = df['both_class0'].astype(int) + df['both_class1'].astype(int)
    nulls = df['null_class0'].astype(int) +df['null_class1'].astype(int)

    ones = ones.to_list()
    zeros = zeros.to_list()
    boths = boths.to_list()
    nulls = nulls.to_list()

    pt.clf()
    pt.style.use('seaborn-white')
    if latex: pt.rc("text", usetex=True)
    fig, ax = pt.subplots(figsize=(10,10))
    
    blue = (0.000000000,0.419607843,1.000000000)
    red = (1.000000000, 0.501960784, 0.000000000)

    pt.ylabel("relative set distribution", fontweight='bold',fontsize=46)

    pt.xlabel(r'significance (Epsilon)', fontweight='bold',fontsize=46)
    if latex: pt.xlabel(r'significance ($\varepsilon$)', fontweight='bold',fontsize=46)

    pt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],fontsize=36)
    pt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],fontsize=36)
    pt.tick_params(length=5, width=2.5)
    
    for axis in ['top', 'bottom', 'left', 'right']: ax.spines[axis].set_linewidth(2.5)
    
    pt.yticks(fontsize=36)
    pt.xticks(fontsize=36)
    pt.tick_params(length=10, width=3)

    total = ones[0] + zeros[0] + boths[0] + nulls[0]

    cum_1 = [float(k)/total for k in ones]
    pt.plot(significances, cum_1, label="{1}", color=blue)
    pt.plot(significances, cum_1, color=blue, lw=3)
    pt.fill_between(significances, cum_1, [0]*len(significances), facecolor=blue, alpha=0.2)
    
    cum_2 = [float(zeros[i])/total + cum_1[i] for i in range(len(significances))] 
    pt.plot(significances, cum_2, label="{0}", color=red)
    pt.plot(significances, cum_2, color=red, lw=3,zorder=1)
    pt.fill_between(significances, cum_2, cum_1, facecolor=red, alpha=0.2)

    cum_3 = [float(nulls[i])/total + cum_2[i] for i in range(len(significances))]
    pt.fill_between(significances, cum_3, cum_2, facecolor=(0.5,0.5,0.5,0.2))

    cum_4 = [float(boths[i])/total + cum_3[i] for i in range(len(significances))]
    pt.fill_between(significances, cum_4, cum_3, facecolor=(230.0/255,230.0/255,250.0/255,0.2))

    optimalSignificance = significances[np.argmax(cum_2)]

    logger.info(f'Number of samples: {total}')

    logger.info(f'Optimal significance: {optimalSignificance}')

    pt.axvline(x = optimalSignificance, color = 'black', linestyle = '--', linewidth=3)

    pt.text(optimalSignificance, 0.5, r"Epsilon", va="center", ha="center", rotation=90, bbox=dict(facecolor="white",ec="black"), fontsize=30)
    if latex: pt.text(optimalSignificance, 0.5, r"$\mathbf{\varepsilon_{opt}}$", va="center", ha="center", rotation=90, bbox=dict(facecolor="white",ec="black"), fontsize=30)

    pt.ylim((0,1))
    pt.xlim((0,1))
    
    # add legend
    handles, labels = ax.get_legend_handles_labels()

    # copy the handles
    handles = [copy.copy(ha) for ha in handles ]
    [ha.set_linewidth(7) for ha in handles ]

    # print('aaa')

    # try:

    fig.savefig(outputfile, bbox_inches='tight', dpi=300, transparent=False)

    # except RuntimeError as e:

        # logger.warning(e)
        # pt.text(optimalSignificance, 0.5, r"Epsilon", va="center", ha="center", rotation=90, bbox=dict(facecolor="white",ec="black"), fontsize=30)
        # pt.xlabel(r'significance (Epsilon)', fontweight='bold',fontsize=46)
        # fig.savefig(outputfile, bbox_inches='tight', dpi=300, transparent=False)



    # print('aaa')

    # exit()
    
