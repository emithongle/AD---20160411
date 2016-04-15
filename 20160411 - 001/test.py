# from libs.features import getFeatureNames, preprocess, feature
#
# def printData(text):
#     print('Text : ', text)
#     for i, j in zip(getFeatureNames(), feature(preprocess(text))):
#         print(i, ' : ', j)
#     print('---------------------------------')
#
# printData('q 10')
# printData('qua 3')

# import execute.test_address_segment as tas
# import execute.test_term_classifier_model as ttc
#
# tas.exc()
# ttc.exc()

from libs.features import *

print(findMaxString('435/5 nguyen van cong, p 3, q.go vap', 0, skip_punctuation))
print(findMaxString('36/6b quang trung, p.10, q. go vap', 0, skip_punctuation))

print(findMaxString('13/8 ấp tay a, x. dong hoa, di an, binh duong', 0, skip_punctuation))

