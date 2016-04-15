from libs import store, models
from libs.features import *
from config import *
import xlsxwriter

modelInfos, modelDict = store.loadAllModel()

groups = models.groupModels(modelInfos, modelDict)

data = [['#', 'Learning Rate', 'Learning Rule', 'N_Iter', 'Avg_Mean_Distance', 'Avg_Var_Distance',
         'alpha', 'H0: Avg_Mean_Distance = 0', 'P(distance < ' + str(delta_threshold) + ')',
         'p(Accept H0: mean_1 = mean_2)']]

tmp = store.loadTermData()
termList = {'X': [i[0] for i in tmp], 'y': [int(i[1]) for i in tmp]}
_X = np.asarray([extractFeatureText(term, getFeatureNames()) for term in termList['X']])

workbook_d = xlsxwriter.Workbook(folder_model + '/' + file_model_details)
store.writeSheet(workbook_d.add_worksheet('GroupInfo'),
    [['Group', 'Learning_Rate', 'Learning_Rule', 'n_Iter']] + \
    [[i,
      g['group-info']['learning_rate'],
      g['group-info']['learning_rule'],
      g['group-info']['n_iter']] for i, g in zip(range(len(groups)), groups)]
)

for i, igroup in zip(range(len(groups)), groups):
    results = models.checkModelConvergence(igroup['models'], _X)

    store.writeSheet(workbook_d.add_worksheet('Group' + str(i)), results['data'])

    data.append([
        i,
        igroup['group-info']['learning_rate'],
        igroup['group-info']['learning_rule'],
        igroup['group-info']['n_iter'],
        results['mean'],
        results['var'],
        alpha,
        results['H0: distance_mean'],
        results['P(diff < delta)'],
        results['p(Accept H0: mean_1 = mean_2)']
    ])

workbook_d.close()
# ================================
workbook = xlsxwriter.Workbook(folder_model + '/' + file_model_result)
store.writeSheet(workbook.add_worksheet('original'), data)
workbook.close()



