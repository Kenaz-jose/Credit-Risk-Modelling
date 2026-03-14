import os
import sys
from creditriskmodelling.entity.artifact_entity import ClassificationMetricArtifact
from creditriskmodelling.exception.exception import CreditRiskModellingException
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,roc_curve

def get_classification_score(y_true,y_pred,y_pred_proba) -> ClassificationMetricArtifact:
    try:

        model_f1_score = f1_score(y_true,y_pred)
        model_recall_score = recall_score(y_true,y_pred)
        model_precision_score = precision_score(y_true,y_pred)
        model_roc_auc_score = roc_auc_score(y_true, y_pred_proba)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        ks_statistic = max(tpr - fpr)

        classification_metric = ClassificationMetricArtifact(f1_score=model_f1_score,
                                                             recall_score=model_recall_score,
                                                             precision_score=model_precision_score,
                                                             roc_auc_score=model_roc_auc_score,
                                                             ks_statistic=ks_statistic)
        return classification_metric
    
    except Exception as e:
        raise CreditRiskModellingException(e,sys)