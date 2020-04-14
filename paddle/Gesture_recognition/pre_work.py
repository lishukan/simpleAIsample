#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
from PIL import Image
import paddle
import numpy as np
from multiprocessing import cpu_count
from paddle.fluid import Linear,Conv2D,Pool2D
from paddle import fluid
def generate_data_file(data_path=None):
    if not data_path:
        data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),'Dataset' )
    character_folders = os.listdir(data_path)
    print(character_folders)


    if(os.path.exists('./train_data.list')):
        os.remove('./train_data.list')
    if(os.path.exists('./test_data.list')):
        os.remove('./test_data.list')
    
    try:
        for character_folder in character_folders:
            with open('./train_data.list', 'a') as f_train:
                with open('./test_data.list', 'a') as f_test:
                    if character_folder == '.DS_Store':
                        continue
                    character_imgs = os.listdir(os.path.join(data_path,character_folder))
                    count = 0 
                    for img in character_imgs:
                        if img =='.DS_Store':
                            continue
                        if count%10 == 0:
                            f_test.write(os.path.join(data_path,character_folder,img) + '\t' + character_folder + '\n')
                        else:
                            f_train.write(os.path.join(data_path,character_folder,img) + '\t' + character_folder + '\n')
                        count +=1
        print('列表已生成')
        return  True
    except:
        import traceback
        traceback.print_exc()
        return False


# 定义训练集和测试集的reader
def data_mapper(sample):
    img, label = sample
    img = Image.open(img)
    img = img.resize((100, 100), Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = img.transpose((2, 0, 1))
    img = img/255.0
    return img, label

def data_reader(data_list_path):
    def reader():
        with open(data_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img, label = line.split('\t')
                yield img, int(label)
    return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), 512)

# #定义DNN网络
# # class MyDNN(fluid.dygraph.Layer):
# #     def __init__(self):
# #         super(MyDNN, self).__init__()
# #         self.hidden1 = Linear(100, 300, act="relu")
# #         self.hidden2 = Linear(300, 300, act="relu")
# #         self.hidden3 = Linear(300, 100, act="relu")
# #         self.hidden4 = Linear(3*100*100, 10, act="softmax")
# #     def forward(self, input):
# #         x = self.hidden1(input)
# #         x = self.hidden2(x)
# #         x = self.hidden3(x)
# #         x = fluid.layers.reshape(x, shape=[-1, 3*100*100])
# #         y = self.hidden4(x)
# #         return y
train_reader = paddle.batch(reader=paddle.reader.shuffle(reader=data_reader('./train_data.list'), buf_size=256), batch_size=32)
# 用于测试的数据提供器
test_reader = paddle.batch(reader=data_reader('./test_data.list'), batch_size=32) 

class MyDNN(fluid.dygraph.Layer):
    def __init__(self,training=True):
        super(MyDNN,self).__init__()
        self.conv1 = Conv2D(num_channels=3,num_filters=32,filter_size=3,act='relu')
        self.pool1 = Pool2D(pool_size=2,pool_stride=2,pool_type='max')
        self.conv2 = Conv2D(num_channels=32, num_filters=32, filter_size=3, act='relu')
        self.pool2 = Pool2D(pool_size=2, pool_stride=2,pool_type='max')
        self.conv3 = Conv2D(num_channels=32, num_filters=64, filter_size=3, act='relu')
        self.pool3 = Pool2D(pool_size=2, pool_stride=2,pool_type='max')
        self.conv4 = Conv2D(num_channels=64, num_filters=128, filter_size=3, act='relu')
        self.pool4 = Pool2D(pool_size=2, pool_stride=2,pool_type='max')
        self.conv5 = Conv2D(num_channels=128, num_filters=256, filter_size=3, act='relu')
        self.fc1 = Linear(input_dim=1024,output_dim=5000,act='relu')
        self.drop_ratiol = 0.5 if training else 0.0
        self.fc2 = Linear(input_dim=5000, output_dim=10)

    def forward(self,input1):
        # 卷积层1 --> 池化层1 --> 卷积层2 --> 池化层2 --> 卷积层3 --> 池化层3 --> 卷积层4 --> 池化层4 --> 卷积层5 --> 全连接网络1 --> 丢弃 --> 全连接网络2
        conv1 = self.conv1(input1)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)
        rs_1 = fluid.layers.reshape(conv5,[conv5.shape[0],-1])
        fc1 = self.fc1(rs_1)
        drop1 = fluid.layers.dropout(fc1,self.drop_ratiol)
        y = self.fc2(drop1)
        return y

def start_train():
    #用动态图进行训练
    with fluid.dygraph.guard():
        model=MyDNN(True) #模型实例化
        model.train() #训练模式
        # opt=fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model.parameters())#优化器选用SGD随机梯度下降，学习率为0.001.
        opt = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameter_list=model.parameters())
        epochs_num=160 #迭代次数
        
        for pass_num in range(epochs_num):
            
            for batch_id,data in enumerate(train_reader()):
                
                images=np.array([x[0].reshape(3,100,100) for x in data],np.float32)
                
                labels = np.array([x[1] for x in data]).astype('int64')
                labels = labels[:, np.newaxis]
                # print(images.shape)
                image=fluid.dygraph.to_variable(images)
                label=fluid.dygraph.to_variable(labels)
                predict=model(image)#预测
                # print(predict)
                sf_predict = fluid.layers.softmax(predict)
                # loss=fluid.layers.cross_entropy(predict,label)
                loss = fluid.layers.softmax_with_cross_entropy(predict, label)
                avg_loss=fluid.layers.mean(loss)#获取loss值

                acc=fluid.layers.accuracy(sf_predict,label)#计算精度
                
                if batch_id!=0 and batch_id%50==0:
                    print("train_pass:{},batch_id:{},train_loss:{},train_acc:{}".format(pass_num,batch_id,avg_loss.numpy(),acc.numpy()))
                
                avg_loss.backward()
                opt.minimize(avg_loss)
                model.clear_gradients()
                
        fluid.save_dygraph(model.state_dict(),'MyDNN')#保存模型

def verify_model():
    #模型校验
    with fluid.dygraph.guard():
        accs = []
        model_dict, _ = fluid.load_dygraph('MyDNN')
        model = MyDNN()
        model.load_dict(model_dict) #加载模型参数
        model.eval() #训练模式
        for batch_id,data in enumerate(test_reader()):#测试集
            images=np.array([x[0].reshape(3,100,100) for x in data],np.float32)
            labels = np.array([x[1] for x in data]).astype('int64')
            labels = labels[:, np.newaxis]

            image=fluid.dygraph.to_variable(images)
            label=fluid.dygraph.to_variable(labels)
            
            predict=model(image)       
            acc=fluid.layers.accuracy(predict,label)
            accs.append(acc.numpy()[0])
            avg_acc = np.mean(accs)
        print(avg_acc)



def final_test(path):
    #读取预测图像，进行预测

    def load_image(path):
        img = Image.open(path)
        img = img.resize((100, 100), Image.ANTIALIAS)
        img = np.array(img).astype('float32')
        img = img.transpose((2, 0, 1))
        img = img/255.0
        print(img.shape)
        return img

    #构建预测动态图过程
    with fluid.dygraph.guard():
        model=MyDNN()#模型实例化
        model_dict,_=fluid.load_dygraph('MyDNN')
        model.load_dict(model_dict)#加载模型参数
        model.eval()#评估模式
        infer_img = load_image(path)
        infer_img=np.array(infer_img).astype('float32')
        infer_img=infer_img[np.newaxis,:, : ,:]
        infer_img = fluid.dygraph.to_variable(infer_img)
        result=model(infer_img)
        #display(Image.open('手势.JPG'))
        print(np.argmax(result.numpy()))


if __name__ == "__main__":
    #verify_model()
    final_test("55.jpg")
