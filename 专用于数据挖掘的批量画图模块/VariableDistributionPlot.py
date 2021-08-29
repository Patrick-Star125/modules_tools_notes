from __future__ import division
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from scipy import stats


class VariableDistribution():
    # input pandas table and feature, return the image of values distribution of this feature
    def __init__(self):
        self.plot_size = self.multiplication()

    def multiplication(self):
        cache = {}
        for i in range(1, 10):
            for j in range(1, i + 1):
                cache[i * j] = [j, i]
        order = sorted(cache)
        a = OrderedDict()
        for i in range(len(order)):
            a[order[i]] = cache[order[i]]
        return a

    def ScatterPlot(self):
        sns.scatterplot()
        pass

    def BoxPlot(self, data, feature=None, save=False, together=False,lines=None):
        '''
        data: pandas table dataset(train_dataset) with atleast one feature
        feature: a list contain few feature string. eg: ['V0','V3','V2']
        save: save image in local folder
        '''
        if together == False:
            if feature == None:
                column = data.columns.tolist()[:]
            else:
                column = feature
            length = len(column)

            for i in self.plot_size:
                if length <= i:
                    cache = i
                    break

            length_p = self.plot_size[cache]
            for i in range(length):
                try:
                    plt.subplot(length_p[0], length_p[1], i + 1)
                    sns.boxplot(y=data[column[i]], orient="v")
                except:
                    print("‘{}’特征数据无法正常画出箱型图,请检查数据是否正常".format(column[i]))
                plt.ylabel(column[i])
        else:
            plt.subplot(1, 1, 1)
            plt.boxplot(x=data.values, labels=data.columns)
            if lines != None:
                plt.hlines([-lines, lines], 0, 40, colors='r')
        plt.show()
        if save == True:
            plt.savefig('VariableBoxPlot.jpg')
            print("图像已保存至VariableBoxPlot.jpg中")
        else:
            print("未自动保存图片")


    def DistributionPlot(self, data, feature=None, save=False):
        '''
        data: pandas table dataset(train_dataset) with atleast one feature
        feature: a list contain few feature string. eg: ['V0','V3','V2']
        save: save image in local folder
        '''
        if feature == None:
            column = data.columns.tolist()[:]
        else:
            column = feature
        length = len(column)

        for i in self.plot_size:
            if length <= i:
                cache = i
                break

        length_p = self.plot_size[cache]
        for i in range(length):
            try:
                plt.subplot(length_p[0], length_p[1], i + 1)
                sns.distplot(data[column[i]], fit=stats.norm)
            except:
                print("‘{}’特征数据无法正常画出分布图,请检查数据是否正常".format(column[i]))
            plt.ylabel(column[i])
        plt.show()
        if save == True:
            plt.savefig('VariableDistributionPlot.jpg')
            print("图像已保存至VariableDistributionPlot.jpg中")
        else:
            print("未自动保存图片")

    def QQPlot(self, data, feature=None, save=False):
        '''
        data: pandas table dataset(train_dataset) with atleast one feature
        feature: a list contain few feature string. eg: ['V0','V3','V2']
        save: save image in local folder
        '''
        if feature == None:
            column = data.columns.tolist()[:]
        else:
            column = feature
        length = len(column)

        for i in self.plot_size:
            if length <= i:
                cache = i
                break

        length_p = self.plot_size[cache]
        for i in range(length):
            try:
                plt.subplot(length_p[0], length_p[1], i + 1)
                stats.probplot(data[column[i]], plot=plt)
            except:
                print("‘{}’特征数据无法正常画出Q-Q图,请检查数据是否正常".format(column[i]))
            plt.ylabel(column[i])
        plt.show()
        if save == True:
            plt.savefig('VariableQQPlot.jpg')
            print("图像已保存至VariableQQPlot.jpg中")
        else:
            print("未自动保存图片")

    def VioliongramPlot(self):
        sns.violinplot()
        pass

    def KDEPlot(self, data, feature=None, save=False):
        '''
        data: pandas table dataset(train_dataset and test_dataset) with atleast one feature
        feature: a list contain few feature string. eg: ['V0','V3','V2']
        save: save image in local folder
        '''
        if feature == None:
            column = data[1].columns.tolist()[:]
        else:
            column = feature
        length = len(column)

        for i in self.plot_size:
            if length <= i:
                cache = i
                break

        length_p = self.plot_size[cache]
        for i in range(length):
            try:
                plt.subplot(length_p[0], length_p[1], i + 1)
                sns.kdeplot(data[0][column[i]], color="Red", shade=True)
                sns.kdeplot(data[1][column[i]], color="Blue", shade=True)
            except:
                print("‘{}’特征数据无法正常画出KDE图,请检查数据是否正常".format(column[i]))
            plt.ylabel(column[i])
        plt.show()
        if save == True:
            plt.savefig('VariableKDEPlot.jpg')
            print("图像已保存至VariableKDEPlot.jpg中")
        else:
            print("未自动保存图片")

    def RegPlot(self, datas, target, feature=None, save=False):
        '''
        data: pandas table dataset(train_dataset) with atleast one feature
        target: the key of label in the dataset
        feature: a list contain few feature string. eg: ['V0','V3','V2']
        save: save image in local folder
        '''
        if feature == None:
            column = datas.columns.tolist()[:]
        else:
            column = feature
        length = len(column)

        for i in self.plot_size:
            if length <= i:
                cache = i
                break

        length_p = self.plot_size[cache]
        for i in range(length):
            try:
                ax = plt.subplot(length_p[0], length_p[1], i + 1)
                sns.regplot(x=column[i], y=target, data=datas, ax=ax,
                            scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3}, line_kws={'color': 'k'})
            except:
                print("‘{}’特征数据无法正常画出线性相关图,请检查数据是否正常".format(column[i]))
            plt.xlabel(column[i])
            plt.ylabel(target)
        plt.show()
        if save == True:
            plt.savefig('VariableRegPlot.jpg')
            print("图像已保存至VariableRegPlot.jpg中")
        else:
            print("未自动保存图片")


    def HeatPlot(self, data, drop=None, save=False, filter=None, k=None, threshold=None):
        '''
        :param data: dataset(train_dataset) with atleast one feature dtpye:pandas table
        :param drop: remove some feature dtype: list of index string
        :param save: decide whether to sava image to file dtype: boolen
        :param filter:
        :param k:
        :param threshold:
        '''
        if drop != None:
            data = data.drop(drop, axis=1)
        data = data.corr()
        try:
            plt.subplot(1, 1, 1)
            sns.heatmap(data, vmax=0.8, square=True, annot=True)
        except:
            print("特征数据无法正常画出相关热力图,请检查数据是否正常")
        plt.show()
        if save == True:
            plt.savefig('VariableHeatPlot.jpg')
            print("图像已保存至VariableHeatPlot.jpg中")

        if filter != None:
            if k == None:
                print("please give an int value to 'k'")
            try:
                cols = data.nlargest(k, filter)[filter].index
                plt.subplot(1, 2, 1)
                sns.heatmap(data[cols].corr(), annot=True, square=True)
            except:
                print("特征数据无法正常画出K-相关热力图,请检查数据是否正常")
            if threshold != None:
                top_corr = data[data[filter].abs() > threshold].index
                plt.subplot(1, 2, 2)
                sns.heatmap(data[top_corr].corr(), annot=True, square=True)
                try:
                    top_corr = data[data[filter].abs() > threshold].index
                    plt.subplot(1, 2, 2)
                    sns.heatmap(data[top_corr].corr(), annot=True, square=True)
                except:
                    print("特征数据无法正常画出阈值热力图,请检查数据是否正常 | 请检查threshold数据是否正确")
        plt.show()
        if save == True:
            plt.savefig('K_VariableHeatPlot.jpg')
            print("图像已保存至K_VariableHeatPlot.jpg中")
        else:
            print("未自动保存图片")


def main():
    pass


if __name__ == '__main__':
    main()
