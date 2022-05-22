## 仿真数据集代码使用说明
前置条件         
1、确认已经安装好Cuda、cudnn等环境；     
2、安装Anaconda；    
3、训练
```
git clone https://github.com/XiangpengZeng/UAV-Action.git
conda create -n envs_name python=3.7
conda activate envs_name
conda install tensorflow-gpu==1.13.1
pip install-U scikit-learn
pip install opencv-contrib-python
cd UAV-Action/Simulation-code
python Simulation-code.py
```
4、多视角融合测试，使用时根据需求选择测试数据
```
cd Test
python Simulation-test.py
```
