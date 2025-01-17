import math

# inspired by https://rdrr.io/cran/auctestr/src/R/auc_compare.R
def get_se_auc(auc, np, nn):
    dp = (np - 1) * ((auc/(2 - auc)) - auc**2)
    dn = (nn - 1) * ((2 * auc**2)/(1 + auc) - auc**2)
    se_auc = math.sqrt((auc * (1 - auc) + dp + dn)/(np * nn))
    return se_auc