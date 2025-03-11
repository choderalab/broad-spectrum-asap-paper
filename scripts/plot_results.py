import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from typing import List, Union, Tuple
import pandas as pd


def calculate_correlation_stats(x, y):
    # calculate correlation stats
    from cinnabar import stats
    from scipy.stats import kendalltau, pearsonr, spearmanr
    a = stats.bootstrap_statistic(x, y, statistic='RMSE')
    ktau = np.round(kendalltau(x, y)[0],5)
    pearson = np.round(pearsonr(x, y)[0],5)
    spearman = np.round(spearmanr(x, y)[0],5)
    return a, ktau, pearson, spearman

def bootstrap_auc(pred_labels: Union[List,np.ndarray], exp_labels: Union[List,np.ndarray], n_bootstraps=500):
    """Function to perform bootstrap sampling and calculate AUC

    Parameters
    ----------
    pred_labels : Union[List,np.ndarray]
        Predicted labels
    exp_labels : Union[List,np.ndarray]
        Actual labels
    n_bootstraps : int, optional
        Number of bootstrap samples, by default 500

    Returns
    -------
    Tuple
        Mean AUC, standard error, lower and upper bounds of the 95% confidence interval
    """
    # Store AUCs
    bootstrapped_aucs = []
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Number of samples
    n_samples = len(pred_labels)
    
    for i in range(n_bootstraps):
        # Generate random indices for bootstrap sampling with replacement
        indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
        # Get the bootstrap samples
        pred_sample = pred_labels[indices]
        exp_sample = exp_labels[indices]
        # Calculate the ROC curve and AUC for the bootstrap sample
        fpr, tpr, _ = metrics.roc_curve(exp_sample, pred_sample)
        auc_score = metrics.auc(fpr, tpr)
        # Store the AUC
        bootstrapped_aucs.append(auc_score)
    bootstrapped_aucs = np.array(bootstrapped_aucs)
    bootstrap_aucs_clean = bootstrapped_aucs[np.logical_not(np.isnan(bootstrapped_aucs))]
    # Calculate confidence intervals
    sorted_aucs = np.sort(bootstrap_aucs_clean)
    ci_lower = np.percentile(sorted_aucs, 2.5)
    ci_upper = np.percentile(sorted_aucs, 97.5)
    # Calculate mean AUC and standard error
    mean_auc = np.mean(bootstrap_aucs_clean)
    standard_error = np.std(bootstrap_aucs_clean) / np.sqrt(n_bootstraps)
    
    return mean_auc, standard_error, ci_lower, ci_upper

def get_roc_curve(ax, 
                  pred_labels,
                  exp_labels, 
                  n_bootstraps=500, 
                  calc_std=True, 
                  legend=False, 
                  pred_type='', 
                  title='',
                  color='tab:blue',
                  box=True):
    
    fpr, tpr, thresholds = metrics.roc_curve(exp_labels, pred_labels)
    roc_auc = round(metrics.auc(fpr, tpr),2)
    if calc_std:
        mean_auc, stdev, ci_lower, ci_upper= bootstrap_auc(pred_labels, exp_labels, n_bootstraps)
        print(f"AUC is {roc_auc}, with bootstrap-sampling is {round(mean_auc,2)}+/-{round(stdev,3)}")
        print(f"95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
    else:
        mean_auc, stdev = roc_auc, 0
    # Plot the ROC curve
    plt.rcParams.update({'font.size': 12})
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the tick labels

    label = ""
    if legend:
        label = pred_type + f" AUC={roc_auc}"
    ax.plot(fpr, tpr, label=label, color=color)
    ax.plot([0, 1], [0, 1], 'k--', )
    ax.set_xlim([0.0, 1.01])
    ax.set_ylim([0.0, 1.01])
    ax.set_xlabel('False Positive Rate',fontsize=14)
    ax.set_ylabel('True Positive Rate',fontsize=14)
    ax.set_title(f'{title}, {pred_type}', fontsize=12)
    if box:
        ax.text(0.62, 0.1, f'AUC={round(mean_auc,2)}+/-{round(stdev,3)}', bbox=dict(alpha=0.1, pad=5), fontsize=12,  ha="center")

    return ax

def calculate_classif_metrics(tn, fp, fn, tp):
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    precision = tp/(tp+fp)
    return sensitivity, specificity, precision

def get_conf_mat(ax, pred, exp, pred_criteria, exp_criteria, pred_type):
    plt.rcParams.update({'font.size': 12})
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    pred_labels = pred >= pred_criteria
    exp_labels = exp >= exp_criteria
    # fig, ax = plt.subplots(layout='constrained', figsize=(5,5))
    cm = metrics.confusion_matrix(exp_labels, pred_labels)
    mcc = metrics.matthews_corrcoef(exp_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()
    sensitivity, specificity, precision = calculate_classif_metrics(tn,fp,fn,tp)
    text = f"MCC={round(mcc,2)}, Sensitivity={round(sensitivity,2)}, \nSpecificity={round(specificity,2)}, Precision={round(precision,2)}"
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'], ax=ax, annot_kws={"size": 16})
    ax.set_xlabel('Predicted inhibitors', fontsize=12, labelpad=2)
    ax.set_ylabel('True inhibitors', fontsize=12)
    ax.set_title(f'signal recov>{exp_criteria}, {pred_type}<{round(pred_criteria,1)}', fontsize=12)

    ax.text(1, -0.48, text, bbox=dict(alpha=0.1, pad=5), fontsize=12,  ha="center", va='bottom')

    return ax


def exp_and_pred_affinity(ic50_df, pred_df, ic50_col='IC50 (µM)', dock_col='', 
                          binder_cond=5.0, outlier_pic50=15, 
                          ):
    # Calculate exp pIC50 and binder label 
    exp_df = ic50_df.copy()
    if 'pIC50' in (ic50_col):
        exp_df['pIC50'] = exp_df[ic50_col]
        exp_df['Affinity (kcal/mol)'] = np.log(10**(-exp_df[ic50_col]))*0.5922 
    else:
        exp_df['pIC50'] = exp_df[ic50_col].apply(lambda x: -np.log10(x*10**(-6)))
        exp_df['Affinity (kcal/mol)'] = np.log(exp_df[ic50_col]*10**(-6))*0.5922 

    exp_df['is_binder'] = exp_df['pIC50']>=binder_cond
    # outliers messed up the statistics
    exp_df = exp_df.loc[(exp_df['pIC50'] < outlier_pic50)]
    all_aff = exp_df
    # Predicted pIC50
    if len(dock_col) > 0:
        lig_affinities = pred_df.copy()
        lig_affinities['pred-pIC50'] = -np.log10(np.exp(lig_affinities[dock_col]/0.5922))
        lig_affinities['is_binder_pred'] = lig_affinities['pred-pIC50']>=binder_cond
        lig_affinities = lig_affinities.loc[(lig_affinities[dock_col] < 0)]
        all_aff = lig_affinities.merge(exp_df[['lig-ID', 'pIC50', 'is_binder', 'Affinity (kcal/mol)']], how = 'inner', on = ['lig-ID'])
    return all_aff

def plot_affinity_compare(exp_df: pd.DataFrame, 
                          pred_df: pd.DataFrame, 
                          target: str,
                          ic50_col: str, 
                          score_col: str,
                          binder_cond: Union[float,int] = 7.0, #50microM
                          prange: List = [-10,-5],
                          yaxis: str = "Predicted Affinity (kcal/mol)",
                          epsilon: float = 0.5,
                          label_pad: float = None,
                          classif_plot: str = 'conf-matrix'):
    """Generate a plot comparing experimental (as IC50s) and predicted affinities (in kcal/mol)

    Parameters
    ----------
    exp_df : pd.DataFrame
        Datframe containing experimental data
    pred_df : pd.DataFrame
        Dataframe containing predicted affinities
    target : str
        Name of the target 
    ic50_col : str
        Name of the column containing IC50 (or pIC50) values
    score_col : str
        Name of the column containing predicted affinities
    binder_cond : Union[float,int], optional
        Minimum pIC50 to classify ligand as inhibitor, by default 7.0
    yaxis : str, optional
        Name of y axis, by default "Predicted Affinity (kcal/mol)"
    epsilon : float, optional
        Parameter defining the size of the shading region, by default 0.5
    label_pad : float, optional
        Padding on label containing correlation stats, by default None, meaning it will be calculated
    classif_plot : str, optional
        Type of plot to be included besides the correlation between 'roc_curve' and 'conf-matrix', by default 'conf-matrix'

    Returns
    -------
    Tuple of pd.DataFrame, Figure, pd.DataFrame, pd.DataFrame
        Dataframe containing all affinity data, figure object, sorted experimental and predicted binders

    Raises
    ------
    NotImplementedError
        If an invalid value is provided for classif_plot
    """
    # Calculate exp pIC50 and binder label 
    all_aff = exp_and_pred_affinity(exp_df, pred_df, ic50_col, score_col, binder_cond)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
 
    ax[0].set_ylabel(yaxis, fontsize=14)
    ax[0].set_xlabel("Experimental (kcal/mol)", fontsize=14)
    #ax.set_facecolor('white')

    x = all_aff['Affinity (kcal/mol)']
    y = all_aff[score_col]

    ax[0].set_xlim(prange[0], prange[1])
    ax[0].set_ylim(prange[0], prange[1])
    ax[0].scatter(x, y, marker="o", linestyle="None", c="tab:blue")
    ax[0].plot(prange, prange, color='black')
    # calculate stats
    a, ktau, pearson, spearman = calculate_correlation_stats(x, y)
    if not label_pad:
        midpoint = (prange[0] + prange[1]) / 2
        label_pad = prange[0] + midpoint / 7
    title = f"RMSE = {a['mle']:.2f}, Kendall's tau = {ktau:.3f}, \nPearson's r = {pearson:.3f}, Spearman r = {spearman:.3f}"  # [95%: {a['low']:.2f}, {a['high']:.2f}]
    ax[0].text((prange[0] + prange[1]) / 2, label_pad, title, bbox=dict(alpha=0.1, pad=5), fontsize=10, ha="center")
    ax[0].set_title(f"{target}") #: RMSE: {title}")
    ax[0].title.set_size(14)
    # Define the shading region
    ax[0].fill_between(np.linspace(prange[0], prange[1], 100), 
                       np.linspace(prange[0], prange[1], 100) - epsilon, 
                       np.linspace(prange[0], prange[1], 100) + epsilon, 
                       color='grey', alpha=0.2, label=f'y=x ± {epsilon}')
    
    ax[1].set_xlabel('Predicted inhibitors', fontsize=14)
    ax[1].set_ylabel('True inhibitors', fontsize=14)
    ax[1].set_title('"Good inhibitors" (IC50<10µM)', fontsize=14)
    if classif_plot == 'conf-matrix':
        ax[1] = get_conf_mat(ax[1], all_aff['is_binder_pred'], all_aff['is_binder'], 
                             binder_cond, binder_cond, 'pIC50', 'pIC50')
    elif classif_plot == 'roc-curve':
        ax[1] = get_roc_curve(ax[1], all_aff['pred-pIC50'], all_aff['is_binder'])
    else:
        raise NotImplementedError("wrong value for classif_plot")
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    nbinders = np.count_nonzero(all_aff['is_binder'])
    sorted_exp = all_aff.sort_values(by=['pIC50'], ascending=False)[:nbinders][['lig-ID', 'pred-pIC50', 'pIC50', 'is_binder_pred']]
    sorted_pred = all_aff.sort_values(by=['pred-pIC50'], ascending=False)[:nbinders][['lig-ID', 'pred-pIC50', 'pIC50', 'is_binder_pred']]
    ab = pd.merge(sorted_exp, sorted_pred, how='inner', on=['lig-ID'])
    print(f'We predicted {len(sorted_exp)}/{len(all_aff)} binders from experiment, out of which {len(ab)} are also predicted from theory')

    return all_aff, fig, sorted_exp, sorted_pred

def plot_from_dict(dict_vals, labels, title, yrange, ylabel, figsize, alpha_dict, label_bars, bwidth):
    proteins = tuple(labels)
    num_meas = len(dict_vals)
    x = np.arange(len(labels))  # the label locations
    num_meas = len(dict_vals)
    width = bwidth  # the width of the bars
    tick_loc = (num_meas//2 - 1) * -1 

    fig, ax = plt.subplots(layout='constrained', figsize=figsize)

    for attribute, measurement in dict_vals.items():
        if alpha_dict is not None:
            alpha = alpha_dict[attribute]
        else:
            alpha = 1
        offset = width * tick_loc
        rects = ax.bar(x + offset, measurement, width, alpha=alpha, label=attribute)
        if label_bars:
            ax.bar_label(rects, padding=3)
        tick_loc += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax. set_ylabel(ylabel,fontsize=18)
    ax.set_title(title, fontsize=18)
    ax.set_xticks(x + width, proteins)
    ax.legend(loc='best', ncols=1)
    ax.set_ylim(yrange[0], yrange[1])
    ax.set_xlim(-0.5, len(labels)+0.2)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14, rotation=90)

    return fig

def plot_bar_scores_in_df(df: pd.DataFrame, 
                          target_labels: List,
                          label_cols: List,
                          sign_list: List, 
                          score_labels: List,
                          title: str, 
                          yrange: List, 
                          ylabel: str, 
                          figsize=(7,5),  
                          label_bars=True,
                          bar_width=0.15):
    """ Generate plot bars for every target scores in a dataframe
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the scores
    target_labels : List
        List with labels for the targets
    label_cols : List
        List with the column names containing the scores
    sign_list : List
        List with the signs of the scores as 1 or -1
    score_labels : List
        List with labels to use for the scores
    title : str
        Title of the plot
    yrange : List
        Range of y axis
    ylabel : str
        Label for y axis
    figsize : tuple, optional
        Size of the figure, by default (7,5)
    label_bars : bool, optional
        Whether to add labels to bars, by default True
    bar_width : float, optional
        Width of the bars, by default 0.15
    """
    def min_max_scale(data):
        min_data = np.min(data)
        max_data = np.max(data)
        return np.round((data-min_data)/(max_data-min_data),2)
    df_scores = df.copy()
    d_vals = {}
    for label_col, sign, label in zip(label_cols, sign_list, score_labels):
        df_scores[label_col] = sign * df_scores[label_col]
        d_vals[label] = tuple(min_max_scale(df_scores[label_col]))

    fig = plot_from_dict(d_vals, target_labels, title, yrange, ylabel, figsize, None, label_bars, bwidth=bar_width)

    return fig
          

def scatter_scores(ax, varx, vary, labelx, labely,
                   variancex=None, variancey=None,
                   xlim=None, ylim=None, point_labels=[],
                   title="", grid=False, hue = None):

    minx = np.min(varx) 
    maxx = np.max(varx)
    miny = np.min(vary) 
    maxy = np.max(vary) 

    # calculate stats
    try:
        a, ktau, pearson, spearman = calculate_correlation_stats(varx, vary)
        title2 = f"{title }Kendall's tau = {ktau:.3f}, Pearson's r = {pearson:.3f}, Spearman r = {spearman:.3f}" + "\n" 
    except: 
        title2 = ""

    colors = {'active':'maroon', 'partial':'tab:orange', 'inactive':'tab:blue',}
    ax.errorbar(varx, vary, xerr=variancex, yerr=variancey, fmt="none", alpha=0.5, capsize=2, elinewidth=1, color="gray")
    if hue is not None:
        c = hue.map(colors).to_numpy() 
        scatter = ax.scatter(varx, vary, c=hue.map(colors), alpha=0.8)
    else: 
        scatter = ax.scatter(varx, vary, c="tab:blue", alpha=0.8)
    ax.set_xlabel(labelx, fontsize=14)
    ax.set_ylabel(labely, fontsize=14)
    ax.set_ylim(miny,maxy)
    ax.set_xlim(minx,maxx)
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    ax.set_title(title2, pad=5, fontsize=9)
    ax.plot([minx, maxx], [miny, maxy], 'k--')  
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    for i, txt in enumerate(point_labels):
        ax.annotate(txt, (varx[i], vary[i]), fontsize=6)
    
    if grid:
        ax.grid(which='major', axis='y', linestyle='--', c='black', alpha=0.5)

    return ax

def plot_score_correlation(df: pd.DataFrame,
                           labels: str, 
                           exp_col: str, 
                           title: str, 
                           yrange: List, 
                           ylabel: str, 
                           xlabel: str, 
                           figsize: Tuple = (7,5), 
                           type: str = 'bars',
):
    """Different plots to visualize the correlation between experimental and predicted scores

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the scores
    labels : str
       Column name containing the labels
    exp_col : str
        Column name containing the experimental scores
    title : str
        Plot title
    yrange : List
        List with range of y axis
    ylabel : str
        Label for y axis
    xlabel : str
        Label for x axis
    figsize : Tuple, optional
        Size of figure, by default (7,5)
    type : str, optional
        Type of plot {'scatter', 'pairplot', 'heatmap', 'bars'}, by default 'bars'

    Returns
    -------
    plt.Figure
        Figure object
    """
    
    if type == 'scatter':
        markers = ['o', 's', '^', 'D', 'v', '*']
        colors = sns.color_palette("tab10", len(df) - 1)
        fig, ax = plt.subplots(layout='constrained', figsize=figsize)
        
        for i, col in enumerate(df.columns[:-3]):
            plt.scatter(df[col], df[exp_col], marker=markers[i], color=colors[i], label=col, alpha=0.5)
            for i, txt in enumerate(df[labels]):
                ax.annotate(txt, (df[col][i], df[exp_col][i]), fontsize=6)
        ax.set_ylabel(ylabel,fontsize=14)
        ax.set_xlabel(xlabel,fontsize=14)
        ax.set_title(title, fontsize=18)
        ax.legend(loc='best', ncols=1)
        ax.set_ylim(yrange[0], yrange[1])
        p = fig

    elif type == 'pairplot':
        p = sns.pairplot(df, diag_kind="kde", markers="o", plot_kws={'alpha':0.7}, hue='Activity')
        p.figure.suptitle(title, fontsize=18, y=1.02)

    elif type == 'heatmap':
        fig, ax = plt.subplots(layout='constrained', figsize=figsize)
        df_numeric = df.select_dtypes(include=['number'])
        corr = df_numeric.corr()
        matrix = np.triu(corr)
        p = sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, 
                linewidths=0.5, fmt=".2f", ax=ax, mask=matrix, annot_kws={"size": 14})
        p.axes.set_xticklabels(p.axes.get_xmajorticklabels(), fontsize = 14)
        p.axes.set_yticklabels(p.axes.get_ymajorticklabels(), fontsize = 14)
        ax.set_title(title, fontsize=18)    
    
    elif type == 'bars':
        df_numeric = df.select_dtypes(include=['number'])
        corr = df_numeric.corr()
        exp_corr = {col: corr.loc[col, exp_col] for col in df_numeric.columns if col in corr.index}
        sorted_corr = dict(sorted(exp_corr.items(), key=lambda item: item[1], reverse=True))
        sorted_corr.pop(exp_col)
        x_labels = list(sorted_corr.keys())  # Column names
        y_values = list(sorted_corr.values())  # Correlation values

        fig, ax = plt.subplots(figsize=(6., 5))
        p = sns.barplot(x=x_labels, y=y_values, ax=ax)
        ax.xaxis.set_ticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=14)  # Rotate x labels 
        ax.axhline(0, color="gray", linestyle="dashed")  # Add reference line at 0
        ax.set_ylim(-0.05,1)
        # ax.set_xlabel("Scoring Functions", fontsize=14)
        ax.set_ylabel(f"Correlation with {exp_col}", fontsize=14)
        ax.set_title(title, fontsize=18)
        # ax.grid(axis='y')
    
    else:
        raise NotImplementedError(f"Plot type {type} not implemented")
    return p.figure

