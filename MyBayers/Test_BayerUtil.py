import os

import pandas as pd
from sklearn.model_selection import train_test_split

from MyBayers import BayerUtil


def load_data(self, filepath):
    '''
    :arg filepath  filepath是数据的路径
    :fun 加载数据：1,青绿,蜷缩,浊响,清晰,凹陷,硬滑,0.697,0.46,是
    :return 加载后的数据
    '''
    file_object = open(filepath, encoding='UTF-8')
    train_data = []
    file_object.readline()  # title
    while 1:
        data = file_object.readline()
        if not data:
            break
        else:
            train_data.append(data)
    file_object.close()
    data = []
    for s in train_data:
        data.append(s.replace('\n', '').split(','))  # 去掉\n和把数据按照’,‘分割再存
    return data

if __name__ == '__main__':

    path = os.getcwd()
    # ------------------------------连续型-----------------------------------------#
    diabetes = pd.read_csv(U"C:/MachineLearn/PurePython/pima-indians-diabetes.csv")
    dia_train, dia_test = train_test_split(diabetes, test_size=0.1)
    v = ['Y', 'Y', 'Y', 'Y', 'Y', 'Y','Y', 'Y']
    model_NBC = BayerUtil.NBBayerUtil(dia_train, v)
    model_NBC.train()
    acc1 = model_NBC.getAccuracy(dia_test)
    print("%.2f" % acc1, "%")

    # -----------------------------离散型------------------------------------------#
    car = pd.read_csv(U"C:/MachineLearn/PurePython/CarEvalution.csv")
    car_train, car_test = train_test_split(car, test_size=0.1)
    v = ['N', 'N', 'N', 'N', 'N', 'N']
    model_NBD = BayerUtil.NBBayerUtil(car_train, v)#NaiveBayesDiscrete.NaiveBayesDiscrete()
    model_NBD.train()
    acc2 = model_NBD.getAccuracy( car_test)
    print()
    print("%.2f" % acc2, "%")
    # -----------------------------混合型------------------------------------------#

    data = pd.read_csv(U"C:/MachineLearn/PurePython/bayes.txt")
    v = ['N','N','N','N','N','N','Y','Y']
    # nb = BayerUtil.NBBayerUtil(data,v)
    # # x1,x2
    # nb.train()
    # test_data = ['青绿', '蜷缩', '清脆', '清晰', '凹陷', '硬滑', 0.697, 0.460]
    #print(nb.predict_v(test_data))

    train, test = train_test_split(data, test_size=0.1)
    print(test)
    nb = BayerUtil.NBBayerUtil(train, v)
    nb.train()
    #print(nb.predict_set(test))
    print(nb.getAccuracy(test))