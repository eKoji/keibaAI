import numpy as np
import lightgbm as lgbm

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
    return f

class MyLGBMClassifier(lgbm.LGBMClassifier):
    """
    custom objective を想定して値を規格化できるように自作classを定義する
    """

    def predict_proba(self, X, *argv, **kwargs):
        proba = super().predict_proba(X, *argv, **kwargs)
        if len(proba.shape) == 2:
            proba = softmax(proba)
        else:
            proba = sigmoid(proba)
            proba[:, 0] = 1 - proba[:, 1]
        return proba

def arr_dev(df_valid_, TARGET_pred, order="infer"):
    mean = np.array(df_valid_[["race_id", TARGET_pred]].groupby("race_id").mean().loc[df_valid_["race_id"], TARGET_pred])
    std = np.array(df_valid_[["race_id", TARGET_pred]].groupby("race_id").std().loc[df_valid_["race_id"], TARGET_pred])
    
    arr = 50 + (df_valid_[TARGET_pred] - mean) / std * 10

    reverse = False
    if order == "infer":
        if np.corrcoef(arr, df_valid_["着順"])[0,1]>0:
            arr = 100 - arr
            reverse = True

    if order == "reverse":
        arr = 100 - arr
        reverse = True
    return arr, reverse

def test(models, _df, _df_encoded, X_cols, TARGET, dev_reverse):
    print(f"        {TARGET}")

    ## set X
    X_test = _df_encoded[X_cols].values

    ## eval
    pred_test = np.zeros(_df_encoded.shape[0])
    for i, model in enumerate(models):
        if "Regressor" not in str(type(model)):
            pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]
        else:
            pred = model.predict(X_test, num_iteration=model.best_iteration_)
        pred_test += pred
    pred_test /= len(models)


    df_test_pred = _df[["着順", "race_id"]]
    df_test_pred[f"{TARGET}_pred"] = pred_test

    if dev_reverse:
        dev_test = arr_dev(df_test_pred, f"{TARGET}_pred", order="reverse")[0]
    else:
        dev_test = arr_dev(df_test_pred, f"{TARGET}_pred", order="foreard")[0]
    
    return pred_test, dev_test