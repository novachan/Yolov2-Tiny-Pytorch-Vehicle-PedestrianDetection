# Ubuntu/Windows-yolov2-pytorch 训练配置

目前网上大家都使用pytorch下yolov2大部分都是https://github.com/marvis/pytorch-yolo2.git（项目已删除）
https://gitee.com/smartcai/pytorch-yolo2#training-on-your-own-data（有人将其删除前克隆了）
cfg.py文件需要修改
https://github.com/soumyadoddagoudar/deepsort_yolov2_pytorch/blob/master/Yolo2/cfg.py 可以
使用的cfg.py文件

本项目是基于这个行人检测进行修改的：https://github.com/emedinac/Pedestrain_Yolov2.



### 准备工作：环境配置

#### 环境配置：

##### 1.安装Anaconda3（或者存在的python3.6环境）

##### 2.运行 Anaconda Prompt (Anaconda3)创建一个 python3.6 的虚拟环境：

```python
conda create -n deepsort python=3.6
```

进入环境：

```python
conda activate deepsort
```

##### 3.安装所需环境：

安装 pytorch==0.4.0：

```python
#这个是linux的：
pip install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m- linux_x86_64.whl
    
#这个是Windows的：
pip install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m-win_amd64.whl
```

安装对应的 torchvision：

```python
pip install torchvision==0.2.2
```

检查是否安装成功，是否能使用 GPU 加速：

```python
 >>>import torch 
 >>>torch.__version__ 
 >>>torch.cuda.is_available()
```

安装 numpy、opencv-python、sklearn：

```python
pip install numpy 
pip install opencv-python 
pip install sklearn
```

**若报错：ModuleNotFoundError: No module named 'sklearn.utils.linear_assignment_'**

**解决办法：降版本0.20.0**

```python
pip3 install -i https://pypi.douban.com/simple scikit-learn==0.20.0
```



### 克隆项目：

克隆本项目

```python
git clone https://github.com/novachan/Yolov2-Tiny-Pytorch-Vehicle-PedestrianDetection.git
    
cd Yolov2-Tiny-Pytorch-Vehicle-PedestrianDetection/
```

下载汽车、行人检测权重：（还在训练就不贴链接了，贴个参考项目的行人检测的权重）

```python
#python detect.py cfg/carperson.cfg carperson.weights data/person.jpg

python detect.py cfg/yolo_person.cfg yolo_person.weights data/person.jpg
```

#### 在VOC数据集下训练：

获取 Pascal VOC 数据

```python
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar

wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar

wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar

tar xf VOCtrainval_11-May-2012.tar

tar xf VOCtrainval_06-Nov-2007.tar

tar xf VOCtest_06-Nov-2007.tar
```

为 VOC 生成标签

运行以下命令以获得完整的测试集，但仅使用人、车检测

```python
python voc_label.py

cat 2007_train.txt 2007_val.txt 2012_*.txt > voc_train.txt

#Windows下将 cat 换成 type
```

运行以下命令以仅拥有人、车类的测试集。

```python
python voc_label_only_class.py

cat 2007_train.txt 2007_val.txt 2012_*.txt > voc_train.txt

#Windows下将 cat 换成 type
```

#### 训练模型:

下载yolov2-tiny-voc.weights：

```python
wget https://pjreddie.com/media/files/yolov2-tiny-voc.weights

```

开始训练：

```python
python train.py cfg/carperson.data cfg/carperson.cfg backup/yolov2-tiny-voc.weights
```



### 可能出现的问题：

问题一：多线程在Windows上的问题（已解决）：

```python
RuntimeError : An attempt has been made to start a new process before thecurrent process has finished its bootstrapping phase. This probably means 

that you are not using fork to start yourchild processes and you have forgotten to use the proper idiomin the main module: 

​			if name__=='main_：

​				freeze_support( ）

The "freeze_support()" line can be omitted if the programis not going to be frozen to produce an executable.
```

```python
num_workers = int(data_options['num_workers'])
```

**将train.py中的num_workers数值改成0**

问题二：在保存权重时报错（已解决）

```python
train.py:240: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead. 
data = Variable(data, volatile=True) 
Traceback (most recent call last): 
File "train.py", line 284, in <module> 
test(epoch) 
File "train.py", line 265, in test 
if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]: RuntimeError: Expected object of type torch.LongTensor but found type torch.FloatTensor for argument #2 'other'
```

尝试将train.py中的：

```python
if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
```

修改成：

```python
if best_iou > iou_thresh and boxes[best_j][6] == torch.tensor(box_gt[6], dtype=torch.long):
```


