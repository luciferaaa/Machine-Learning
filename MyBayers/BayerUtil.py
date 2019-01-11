# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

class NBBayerUtil():
    def __init__(self,traindata,vect_type_indicator,lamda=1):
        """lamda 算子"""
        self.lamda = lamda
        self.traindata = traindata #数据集必须前面是离散特征,
        self.vect_type_indicator = vect_type_indicator # Y,continue,N,discrete

        self.ck_counts = self.traindata["label"].value_counts()  # 训练样本中类为ck的数量集合,Series(index,count)
        self.ck_name = np.array(self.ck_counts.index)  # 训练样本中类ck名称集合
        self.ds_num = len(self.traindata)  # 训练样本总数N
        self.lab_num = len(self.ck_counts)  # 类的个数K
        self.lab_con_arr = []
        self.lab_con_arr_index = []
        self.lab_dis_arr = []
        self.lab_dis_arr_index = []
        for index,val in enumerate(self.vect_type_indicator):
            if val=='Y': #
                self.lab_con_arr.append(self.traindata.columns.values[index])
                self.lab_con_arr_index.append(index)
                pass
            else:
                self.lab_dis_arr.append(self.traindata.columns.values[index])
                self.lab_dis_arr_index.append(index)
                pass
        pass

    # 计算先验概率
    def calPriorProb(self):
        self.ck_PriorProb = []
        for i in range(self.lab_num):
            cx_PriorProb = (self.ck_counts[i] + self.lamda) / (self.ds_num + self.lab_num * self.lamda)
            self.ck_PriorProb.append(cx_PriorProb)
        pass

    # 计算条件概率
    def calCondProb(self):
        names = locals()  # 使用动态变量
        self.CondProb = []  # 存储所有类别的所有特征取值的条件概率
        self.feat_value = []  # 所有特征取值列表

        # 对于每一类别的数据集
        for i in range(len(self.ck_name)):
            names['Q%s' % i] = self.traindata[self.traindata["label"] == self.ck_name[i]]  # 按类别划分数据集
            names['ConProbC%s' % i] = []  # 定义动态变量，表示各类别中所有特征取值的条件概率集合
            #feature_arr = self.datasource.columns.tolist()[0:len(self.datasource.columns) - 1]  # 获取训练数据集特征集

            # 对于每一个特征求该特征各个取值的条件概率
            for feature in self.lab_dis_arr:
                names['Q%s' % feature] = []  # 定义动态变量，表示某个类别的某个特征的所有取值条件概率
                # 对于某个特征的所有可能取值求条件概率
                for value in self.traindata[feature].value_counts().index.tolist():
                    # 生成所有特征取值列表
                    if value not in self.feat_value:  # 如果这个取值不在列表中，则加入这个取值
                        self.feat_value.append(value)
                    # 这里用了拉普拉斯平滑，使得条件概率不会出现0的情况
                    # 如果某个类的某个特征取值在训练集上都出现过，则这样计算
                    if value in names['Q%s' % i][feature].value_counts():
                        temp = (names['Q%s' % i][feature].value_counts()[value] + self.lamda) / (
                                names['Q%s' % i][feature].value_counts().sum() + len(
                                names['Q%s' % i][feature].value_counts()) * self.lamda)
                    # 如果某个类的某个特征取值并未在训练集上出现，为了避免出现0的情况，分子取1(即lamda平滑因子，取1时为拉普拉斯平滑)
                    else:
                        temp = self.lamda / (names['Q%s' % i][feature].value_counts().sum() + len(
                            names['Q%s' % i][feature].value_counts()) * self.lamda)

                    # 将求得的特征取值条件概率加入列表
                    names['Q%s' % feature].append(temp)
                    pass

                # 将得到的某个类别的某个特征的所有取值条件概率列表加入某个类别中所有特征取值的条件概率集合
                names['ConProbC%s' % i].extend(names['Q%s' % feature])
                pass

            # 将某个类别中所有特征取值的条件概率集合加入所有类别所有特征取值的条件概率集合
            self.CondProb.append(names['ConProbC%s' % i])
        # 将所有特征取值列表也加入所有类别所有特征取值的条件概率集合(后面用来做columns--列索引)
        self.CondProb.append(self.feat_value)

        # 用类别名称的集合来生成行索引index
        index = self.ck_name.tolist()
        index.extend(['other'])  # 此处由于我最后一行是feat_value，后面会删掉，因此在行索引上也多加一个，后面删掉
        # 将所有类别所有特征取值的条件概率集合转换为DataFrame格式
        self.CondProb = pd.DataFrame(self.CondProb, columns=self.CondProb[self.lab_num], index=index)
        self.CondProb.drop(['other'], inplace=True)

    # 获取训练集每个特征的均值和方差以及类标签的取值集合
    def getMeanStdLabel(self):
        # 按类别划分数据
        names = locals()
        for i in range(len(self.ck_name)):
            names['c%s' % i] = self.traindata[self.traindata["label"] == self.ck_name[i]]
        # 按类别对每个属性求均值和方差
        c_mean = []
        c_std = []

        for j in range(len(self.ck_name)):
            names['mc%s' % j] = []
            names['sc%s' % j] = []
            #for k in range(num_feature):
            for k in self.lab_con_arr:
                # print(type(names['c%s' % j]))
                names['mc%s' % j].append(np.mean(names['c%s' % j]['%s' % k]))
                names['sc%s' % j].append(np.std(names['c%s' % j]['%s' % k], ddof=1))

        #for x in range(len(label_arr)):
        for x in range(len(self.ck_name)):
            c_mean.append(names['mc%s' % x])
            c_std.append(names['sc%s' % x])
            names['arr_c%s' % x] = np.array(names['c%s' % x])
        return c_mean, c_std #, label_arr
        pass

    # 计算高斯概率密度函数
    def calcuGaussProb(self, x, mean, stdev):

        mean = mean.astype('float64')
        stdev = stdev.astype('float64')
        exponent = np.exp(-(np.power(x - mean, 2)) / (2 * np.power(stdev, 2)))
        GaussProb = (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent
        return GaussProb

    # 计算连续数据所属类的概率
    def calcuClassProbCon(self, arr, cx_mean, cx_std):
        cx_probabilities = 1
        for i in self.lab_con_arr_index:
            cx_probabilities *= self.calcuGaussProb(arr[i], cx_mean[i-self.lab_con_arr_index[0]], cx_std[i-self.lab_con_arr_index[0]])
        return cx_probabilities
        pass

    def train(self):
        self.calPriorProb()  # 获取先验概率
        self.calCondProb()  # 获取条件概率
        self.cmean, self.cstd = self.getMeanStdLabel()
        pass

    def predict_v(self,testdata):

        self.ClassTotalProb = []  # 初始化各类别总概率列表
        bestprob = -1  # 初始化最高概率
        bestfeat = ''  # 初始化最可能类别

        for feat in self.ck_name:
            pp = self.ck_PriorProb[self.ck_name.tolist().index(feat)]  # pp为先验概率
            cp = 1  # 初始化条件概率
            for value in self.feat_value:
                if value in testdata:
                    cp = cp * self.CondProb[value][feat]  # 计算各特征取值的条件概率之积

            pc = 0
            for i in range(len(self.cmean)):
                cx_mean = self.cmean[i]  # x类的均值
                cx_std = self.cstd[i]  # x类的方差
            # print(testData)
                pc = self.calcuClassProbCon(testdata, cx_mean, cx_std)  # 将计算得到的各类别概率存入列表
            TotalProb = pp * cp * pc  # 条件概率之积与先验概率相乘
            self.ClassTotalProb.append(TotalProb)

        bestLabel, bestProb = None, -1  # 初始化最可能的类和最大概率值

        # for i in range(len(prob)):  # 找到所有类别中概率值最大的类
        #     if prob[i] > bestProb:
        #         bestProb = prob[i]
        #         bestLabel = self.label_array[i]

        # 找到最可能类别和其概率
        for i in range(len(self.ck_name)):
            if self.ClassTotalProb[i] > bestprob:
                bestprob = self.ClassTotalProb[i]
                bestfeat = self.ck_name[i]
        return (bestprob, bestfeat)
        pass

    def predict_set(self,testdata):

        self.prediction = []
        self.testdata = np.array(testdata)
        for i in range(len(self.testdata)):
            result, proby = self.predict_v(self.testdata[i])
            self.prediction.append((result,proby))
        return self.prediction
    pass

    # 计算预测准确度
    def getAccuracy(self,  testdata):
        num = 0
        realFeat = testdata.label.tolist()
        for i in range(len(testdata)):
            temp = testdata.iloc[i][0:len(testdata.columns) - 1]
            predProb, predFeat = self.predict_v(temp)
            print(predProb, predFeat, realFeat[i])
            if (realFeat[i] == predFeat):
                num = num + 1
        acc = num / float(len(realFeat)) * 100
        return acc