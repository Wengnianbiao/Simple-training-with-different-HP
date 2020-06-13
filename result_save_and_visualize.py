#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2020/6/10 9:47
# @Author: Biao
# @File: result_save_and_visualize.py
import os
import numpy as np
import matplotlib.pyplot as plt


class RetSAndV(object):
    def __init__(self,epoch,name_txt,bestacc=None,file='ret',dict_loss_acc=None):
        """
        对每一个epoch训练和测试的损失和精度结果进行可视化
        :param epoch:
        :param bestacc: test中最高精度
        :param name_txt: 保存结果的txt文件
        :param file:
        :param dict_loss_acc: 训练和测试结果的txt文件
        """
        self.epoch = epoch
        self.bestacc = bestacc
        self.file = file
        self.cwdpath = os.getcwd()
        self.name_txt = name_txt
        self.dict_loss_acc = dict_loss_acc

    def save_ret(self):
        try:
            file = os.path.join('.',self.file)
            if not os.path.exists(file):
                os.makedirs(file)
            os.chdir(file)
            with open(self.name_txt,'w') as f:
                for key in self.dict_loss_acc:
                    # 使用切片操作不会引发异常
                    f.write(str(key)+'='+str(self.dict_loss_acc[key])[1:-1]+'\n')

        except FileNotFoundError as e:
            raise e
        else:
            print('Data has been stored successfully!')
        finally:
            os.chdir(self.cwdpath)

    def visualize_ret(self,**kwargs):
        try:
            dict_lossacc = self.dict_loss_acc
            # 如果不是在训练阶段，则无法生成dict_loss_acc字典,就直接从txt文件中读取数据
            if dict_lossacc is None:
                
                os.chdir(os.path.join(self.cwdpath,self.file))
                dict_lossacc = {}
                with open(self.name_txt,'r') as f:
                    for line in f.readlines():
                        value = list(map(float,line.split(':',1)[1].strip().split(',')))
                        dict_lossacc[line.split(':',1)[0]] = value

            # ploting
            fig = plt.figure(1)
            main_title = 'HP:'
            if kwargs is not None:
                for key,val in kwargs.items():
                    main_title += str(key)+':'+str(val)+' '

            x = np.arange(1,self.epoch+1).astype(dtype=np.str)
            ax1 = plt.subplot(211)
            plt.plot(x,dict_lossacc['train_loss'], color='green', marker='o', label='train_loss')
            plt.plot(x, dict_lossacc['test_loss'], color='red', marker='*', label='test_loss')
            ax1.set_title(main_title,fontsize=10)
            plt.xlabel('epoch')
            plt.legend()

            ax2 = plt.subplot(212)
            plt.plot(x,dict_lossacc['train_acc'], color='green', marker='o', label='train_acc')
            plt.plot(x, dict_lossacc['test_acc'], color='red', marker='*',label='train_acc')
            ax2.set_title('Accuracy'+'(best test acc:'+str(self.bestacc)+')',fontsize=10)
            plt.xlabel('epoch')
            plt.ylim(0.0,1.0)
            plt.legend()

            plt.tight_layout()
            # Save figure
            fig_folder = os.path.join(self.cwdpath, 'figure')
            if not os.path.exists(fig_folder):
                os.makedirs(fig_folder)
            plt.savefig(os.path.join(fig_folder, self.name_txt[0:-4] + '.jpg'))

            plt.show()
        # txt file does not exist.
        except FileNotFoundError as e:
            raise e
        else:
            print('The graphics are finished successfully!')
        # keep the path the same after processing.
        finally:
            os.chdir(self.cwdpath)


if __name__ == '__main__':
    r = RetSAndV(25,'data10.txt')

    r.visualize_ret(epoch=25,bestacc=0.2,classes=10)
