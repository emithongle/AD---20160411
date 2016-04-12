__author__ = 'Thong_Le'

import json
import datetime

def readFile(file):
    strList = []
    infile = open(file, encoding="utf-8")
    for line in infile:
        strList.append(line)
    return strList

def loadJson(file):
    try:
        return json.loads(''.join(readFile(file + '.json')))
    except:
        return None

class TimeManage():
    def __init__(self):
        self.time = ''

    def setTime(self, _time):
        self.time = _time

    def getTime(self):
        return self.time

timeManage = TimeManage()

# ================================

folder_datasource = '0. Data-source'
files_dictionary = 'dictionary' # json

folder_data = '1. Data'
files_data = ['name.txt', 'address.txt', 'phone.txt']

folder_preprocessing = '2. Preprocessing'
files_preprocessed = ['preprocessed_name', 'preprocessed_address', 'preprocessed_phone'] # csv

folder_features = '3. Features'
files_features = ['name_features', 'address_features', 'phone_features'] # csv
files_traintest = ['training_data', 'testing_data']

folder_model = '4. Model'
file_model = None # '20160329_150346'

folder_test = '5. Test'
file_full_address_test = 'testdata.txt'
file_term_test = 'termdata' # csv

folder_results = '6. Results'
file_segment_address_result = 'full_address_results.xlsx'
file_term_classify_result = 'term_results.xlsx'
file_model_result = 'test_model_results.xlsx'

folder_running_logs = 'running_logs'
file_log = 'logs.xlsx' # csv

# ==============================================

# number of executive run.py
nrun = 4

tmp = loadJson(folder_datasource + '/' + files_dictionary)
nameTermSet = tmp['name-term-set']
addressTermSet = tmp['address-term-set']
phoneTermSet = tmp['phone-term-set']

asi = tmp['ascii']
unic = tmp['unicode']
upchars = tmp['upper-characters']

feature_func = 'feature'
preprocessing_func = 'preprocessing'


# Preprocessing
bpreprocessing = True
preprocessing_name = {'convert unicode to ascii': True, 'convert to lower': True,
                      'remove break line': True, 'remove multiple spaces': True,
                      'trim "space" and ","': True, 'space after punctuation': True}

# Features
feature_list = [
    # ('length', True),
    # ('#ascii', True),
    # ('#digit', True),
    # ('#punctuation', True),

    ('#ascii/(#ascii+#digit+#punctuation)', True),
    ('#digit/(#ascii+#digit+#punctuation)', True),
    ('#digit/#ascii', True),

    # ('%ascii', True),
    # ('%digits', True),

    ('%kwName', True),
    ('%kwAddress', True),
    ('%kwPhone', True),

    ('%max_digit_skip_0', True),
    ('#max_digit_skip_0', False),
    ('%max_digit_skip_0_1', False),
    ('#max_digit_skip_0_1', False),

    ('first_character_ascii', True),
    ('first_character_digit', True),
    ('last_character_ascii', True),
    ('last_character_digit', True),

    ('b+', False),
    ('b/\\', False)
]

# Model
model_type = 'Neuron Network'
model_target = 'Classify Name/Address/Phone'

model_config = {
    'layers': [(100, 'Sigmoid'), (3, 'Softmax')],
    'learning_rate': 0.01,
    'learning_rule': 'adagrad',
    'n_iter': 3000
}
model_config_file = 'model_config'


# Templates
template_rm_filters = {
    'Phone: #digit < 8': False,
    'Phone: 2 * _%ascii < _%digit': False,
    'Phone: _%ascii > 0 & %kwPhone = 0': False,
    'Phone: first_character_type != digit': False,

    'Name: #ascii < 5': False,
    'Name: _%digit > 0': False,
    'Name: first_character_type != ascii': False
}


# Testing Model
standard_data = True
nTesting = 10


skip_punctuation = ' .'
rm_preprocessed_punctuation = """ ,"""
