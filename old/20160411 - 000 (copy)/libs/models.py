__author__ = 'Thong_Le'

from sknn.mlp import Classifier, Layer
import numpy as np
from config import model_type, model_config, delta_threshold, alpha
from scipy.stats import norm
import math

def buildClassifer(name='Neuron Network'):
    model = None
    if (name=='Neuron Network'):
        model = Classifier(
            layers=[Layer(model_config['layers'][i][1], units=model_config['layers'][i][0])
                        for i in range(len(model_config['layers']))],
            learning_rule=model_config['learning_rule'],
            learning_rate=model_config['learning_rate'],
            n_iter=model_config['n_iter']
        )
    return model

def modelDetails():
    if (model_type == 'Neuron Network'):
        return str(len(model_config['layers'])) + ' layers: [' + \
                ', '.join([str(n_unit) + '-' + act_func for (n_unit, act_func) in model_config['layers']]) + \
                '], learning_rate: ' + str(model_config['learning_rate']) + ', learning_rule: ' + model_config['learning_rule'] + \
                ', n_iterator: ' + str(model_config['n_iter'])

# def checkSimilarModel(mi1, mi2):
#     return False

def groupModels(modelInfos, models):
    groups = []

    for modelInfo in modelInfos:
        group_info = {
            'learning_rate': modelInfo['model']['config']['learning_rate'],
            'learning_rule': modelInfo['model']['config']['learning_rule'],
            'n_iter': modelInfo['model']['config']['n_iter']
        }

        flag = True
        for igroup in groups:
            if (igroup['group-info'] == group_info):
                flag = False
                igroup['models'][modelInfo['name']] = models[modelInfo['name']]

        if (flag):
            groups.append({
                'group-info': group_info,
                'models': { modelInfo['name'] : models[modelInfo['name']] }
            })

    return groups

def distantProb(prob_1, prob_2):
    tmp = np.power(prob_1 - prob_2, 2)
    return tmp.mean(), tmp.var()

def test2Samples(prob_1, prob_2):

    return True

def checkModelConvergence(models, _X):
    probX = {mkey: models[mkey].predict_proba(_X) for mkey in models}

    data = np.asarray([[]] * (_X.shape[0] + 2))

    tmp = ['', '#'] + list(range(_X.shape[0]))
    data_name = np.asarray(tmp).reshape(len(tmp), 1)
    data_address = data_name
    data_phone = data_name
    for mkey in probX:
        data_name = np.append(
            data_name,
            np.append(
                np.asarray([[mkey], ['Name']]),
                probX[mkey][:, 0].reshape(_X.shape[0], 1),
                axis=0
            ),
            axis=1
        )

        data_address = np.append(
            data_address,
            np.append(
                np.asarray([[mkey], ['Address']]),
                probX[mkey][:, 1].reshape(_X.shape[0], 1),
                axis=0
            ),
            axis=1
        )

        data_phone = np.append(
            data_phone,
            np.append(
                np.asarray([[mkey], ['Phone']]),
                probX[mkey][:, 2].reshape(_X.shape[0], 1),
                axis=0
            ),
            axis=1
        )

    data = np.append(data, data_name, axis=1)
    data = np.append(data, np.asarray([['']] * (_X.shape[0] + 2)), axis=1)
    data = np.append(data, data_address, axis=1)
    data = np.append(data, np.asarray([['']] * (_X.shape[0] + 2)), axis=1)
    data = np.append(data, data_phone, axis=1)

    probX_key = list(probX.keys())
    nmodels = len(probX_key)

    d, v, c, p = 0, 0, 0, 0
    for i in range(nmodels - 1):
        for j in range(i + 1, nmodels):
            td, tv = distantProb(probX[probX_key[i]], probX[probX_key[j]])
            d += td
            v += tv
            c += 1 if test2Samples(probX[probX_key[i]], probX[probX_key[j]]) else 0
            p += sum([1
                        for i in np.power(probX[probX_key[i]] - probX[probX_key[j]], 2).sum(axis=1).reshape(-1).tolist()
                            if (i < delta_threshold)
                      ])

    t = d / math.sqrt(v)
    tt = norm.isf(alpha / 2)
    if (nmodels == 2):
        return {
            'data': data,
            'mean': d,
            'var' : v,
            'H0: distance_mean': 'Accept' if abs(t) < tt else 'Reject',
            'P(diff < delta)': p / _X.shape[0],
            'p(Accept H0: mean_1 = mean_2)': c
        }

    return {
        'data': data,
        'mean': d / ((nmodels - 1) * (nmodels - 2)),
        'var' : v / ((nmodels - 1) * (nmodels - 2)),
        'H0: distance_mean': 'Accept' if abs(t) < tt else 'Reject',
        'P(diff < delta)': p / (_X.shape[0] * (nmodels) * (nmodels - 1) / 2),
        'p(Accept H0: mean_1 = mean_2)': c / ((nmodels) * (nmodels - 1) / 2)
    }
