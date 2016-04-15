__author__ = 'Thong_Le'

from sknn.mlp import Classifier, Layer

from config import model_type, model_config


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