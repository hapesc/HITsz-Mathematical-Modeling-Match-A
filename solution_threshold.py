import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import leastsq

#   import data
path='meta_data/Data5.csv'  #   路径记得修改
raw_data=pd.read_csv(path)
raw_data_M=np.array(raw_data)
Time=raw_data_M[:,0]

#   滑动平均滤波
def filter1d(raw_data,T=125):
    data_p=np.zeros(raw_data.shape)
    
    for k in range(len(raw_data)):
        if k< (T-1)/2  :
            data_p[k]=np.mean(raw_data[k:int(k+(T-1)/2+1)])
        elif len(raw_data)-1-k<(T-1)/2:
            data_p[k]=np.mean(raw_data[int(k-(T-1)/2):k])

        else:
            data_p[k]=np.mean(raw_data[int(k-(T-1)/2):int(k+(T-1)/2+1)],0)
    return data_p



#   先平滑数据,再计算
#   加大窗口
T=125     #   阈值法的窗口设为125,峰值法设为11
data_p=np.zeros(raw_data_M.shape)
data_p[:,0]=raw_data_M[:,0]
for k in range(len(raw_data_M[:,0])):
    if k< (T-1)/2 or len(raw_data_M[:,0])-1-k<(T-1)/2:
        data_p[k,1:]=raw_data_M[k,1:7]
    else:
        data_p[k,1:]=np.mean(raw_data_M[int(k-(T-1)/2):int(k+(T-1)/2+1),1:7],0)

E_acc=(data_p[:,1]**2 + data_p[:,2]**2 + data_p[:,3]**2)
E_omega=(data_p[:,4]**2 + data_p[:,5]**2 + data_p[:,6]**2)
#   进行归一化
def normalization(data):
    data=2*(data-data.min())/(data.max()-data.min())-1
    return data
E_acc=normalization(E_acc)
E_omega=normalization(E_omega)

#   设立阈值,提取手势区间
threshold=filter1d(E_acc,T=5000)*2
g_point=np.array(np.where(E_acc>threshold)).flatten()
t_point=np.array(np.where(E_acc<=threshold)).flatten()
g_point2=pd.Series(g_point)
g_point2.to_csv('g_point2.csv')
t_point2=pd.Series(t_point)
t_point2.to_csv('t_point2.csv')

#########################################################################
#另一种平滑方法

#归一化
def normalization(data):
    _range = np.max(abs(data))
    return data / _range

#滑动平滑
def move_avg(data,window_length,mode="same"):
    return(np.convolve(data, np.ones((window_length,))/window_length, mode=mode))


Data=pd.read_csv(path)

ax=Data.iloc[:,1:2]
ay=Data.iloc[:,2:3]
az=Data.iloc[:,3:4]
wx=Data.iloc[:,4:5]
wy=Data.iloc[:,5:6]
wz=Data.iloc[:,6:7]
Time=Data.iloc[:,:1]

ax=np.array(ax)
ay=np.array(ay)
az=np.array(az)
wx=np.array(wx)
wy=np.array(wy)
wz=np.array(wz)
for x in np.nditer(ax, op_flags=['readwrite']): 
    x[...] = x * x
for x in np.nditer(ay, op_flags=['readwrite']): 
    x[...] = x * x
for x in np.nditer(az, op_flags=['readwrite']): 
    x[...] = (x - 1) * (x - 1)
for x in np.nditer(wx, op_flags=['readwrite']): 
    x[...] = x * x
for x in np.nditer(wy, op_flags=['readwrite']): 
    x[...] = x * x
for x in np.nditer(wz, op_flags=['readwrite']): 
    x[...] = x * x
Time=np.array(Time)


sum_data_a = ax + ay + az
sum_data_w = wx + wy + wz

Window = 125
sum_data_a = move_avg(sum_data_a.flatten(),Window)
sum_data_a = normalization(sum_data_a)
sum_data_w = move_avg(sum_data_w.flatten(),Window)
sum_data_w = normalization(sum_data_w)

#ans = np.where( sum_data_a > sum_data_w , sum_data_a , sum_data_w )
ans=sum_data_a
flip = move_avg(ans,5000)
for x in np.nditer(flip, op_flags=['readwrite']): 
    x[...] = x * 2

Time=Time.flatten()

gesture_point=np.array(np.where(ans>flip)).flatten()
transition_point=np.array(np.where(ans<=flip)).flatten()
print(gesture_point.shape)
g_point1=pd.Series(gesture_point)
t_point1=pd.Series(transition_point)
g_point1.to_csv('g_point1.csv')
t_point1.to_csv('t_point1.csv')
#######################################################





#   import another data
g_point1=pd.read_csv('g_point1.csv')
t_point1=pd.read_csv('t_point1.csv')
g_point1=np.array(g_point1.iloc[:,1])
t_point1=np.array(t_point1.iloc[:,1])

#   取并集
g_point_final=np.union1d(g_point1,g_point2)
t_point_final=np.setdiff1d(np.arange(len(E_acc)),g_point_final)
startpoint_g=list([g_point_final[0]])
endpoint_g=list()

#   进行分割
for k in range(1,len(g_point_final)-1):
    if g_point_final[k]-g_point_final[k-1]>1:
        startpoint_g.append(g_point_final[k])
        endpoint_g.append(g_point_final[k-1])
endpoint_g.append(g_point_final[-1])
endpoint_g=np.array(endpoint_g)
startpoint_g=np.array(startpoint_g)
action_label_2=np.stack((startpoint_g,endpoint_g),1)

#   删除错误片段
#   distance根据data不同进行调整
distance=150    #控制区间长度
distance2=250   #控制距离

action_label_3=np.delete(action_label_2,np.where((action_label_2[:,1]-action_label_2[:,0])<distance),0)
errorlist=[]
for k in range(10,len(action_label_3)-1):
    if (action_label_3[k,0]-action_label_3[k-1,1])<distance2 or (action_label_3[k+1,0]-action_label_3[k,1]<distance2):
        errorlist.append(k)
errorlist=np.array(errorlist)
#   在除去过渡区间的同时,也把与它相邻的区间除去了
#   通过观察可以发现,与手势区间相比,过渡区间与左右区间的距离里更小
#   因此可以用这个方法来判断出两个彼此相邻的区间中真正的过渡区间
errorlist2=[]
for k in range(len(errorlist)-1):
    if errorlist[k+1]-errorlist[k]==1:
        a=errorlist[k]
        b=errorlist[k+1]
        #   比较距离,删去errorlist中距离大的那个数值
        dis_a=(action_label_3[a,0]-action_label_3[a-1,1]) + (action_label_3[a+1,0]-action_label_3[a,1])
        dis_b=(action_label_3[b,0]-action_label_3[b-1,1]) + (action_label_3[b+1,0]-action_label_3[b,1])
        if dis_a < dis_b:
            errorlist2.append(k+1)
        else:
            errorlist2.append(k)


errorlist=np.delete(errorlist,errorlist2)
action_label_4=np.delete(action_label_3,errorlist,0)



#   导出action_label   记得改文件名
pd.DataFrame(action_label_4,columns=['start','end']).to_csv('Output-data/gesture_label_predict(data5).csv')


#   接下来确定过渡区间
#   首先获得原始区间数据
transition_label=[]
for k in range(1,len(action_label_4)-1):
    transition_label.append([action_label_4[k,1]+1,action_label_4[k+1,0]-1])
    
transition_label=np.array(transition_label)

#   对每个区间进行精确定位
#   经过分析,平滑和归一化后的绝大多数过渡区间具有如下特点:
#   1)  合加速度曲线呈下降趋势
#   2)  合角速度先递增,再递减,具有单峰
#   绝大多数区间至少满足其中一个特征

#   求导,根据斜率判断

slope_a=np.gradient(E_acc,0.01)
slope_omega=np.gradient(E_omega,0.01)
s_o=filter1d(slope_omega,T=99)
s_a=filter1d(slope_a,99)
#   求二阶导
slope_a2=np.gradient(slope_a,0.1)
slope_o2=np.gradient(s_o,0.01)

def func(p,x):
    k,b=p
    return k*x+b

def costfunc(p,x,y):
    return func(p,x)-y

p0=[1,0]    

predict_interval=np.array([0])
for k in range(len(transition_label)):
    begin=transition_label[k,0]
    end=transition_label[k,1]
    W=50
    n=int((end-begin)/W)+1
    list_slope_o=[]
    list_slope_a=[]

    X=np.zeros((n,W),dtype=int)
    cnt=end+1-(begin+(n-1)*W)
    for i in range(n):
        X[i]=np.arange(begin+i*W,begin+(i+1)*W)
    
    for i in range(n):
        #   拟合直线
        para_o=leastsq(costfunc,p0,args=(X[i],E_omega[X[i]]))
        para_a=leastsq(costfunc,p0,args=(X[i],E_acc[X[i]]))
        list_slope_o.append(para_o[0][0])
        list_slope_a.append(para_a[0][0])
    if n<=3:
        for i in range(n-1):
            predict_interval=np.append(predict_interval,X[i])
        predict_interval=np.append(predict_interval,X[n-1][:cnt])
    else:
        list_a=np.array(list_slope_a)
        i=list_a.argmax()
        if n-i<=3:
            for t in range(i,n):
                predict_interval=np.append(predict_interval,X[t])
            predict_interval=np.append(predict_interval,X[n-1][:cnt])
        else:
            for t in range(i,i+3):
                predict_interval=np.append(predict_interval,X[t])

predict_interval=np.unique(predict_interval)
predict_interval=np.delete(predict_interval,0)

#分割区间
startpoint_t=list([predict_interval[0]])
endpoint_t=[]
for k in range(1,len(predict_interval)-1):
    if predict_interval[k]-predict_interval[k-1]>1:
        startpoint_t.append(predict_interval[k])
        endpoint_t.append(predict_interval[k-1])
endpoint_t.append(predict_interval[-1])
startpoint_t
startpoint_t=np.array(startpoint_t)
endpoint_t=np.array(endpoint_t)
non_gesture_label_2=np.stack((startpoint_t,endpoint_t),axis=1)

non_gesture_label_3=non_gesture_label_2

#导出数据
output_label=np.append(action_label_4,non_gesture_label_3,axis=0)
output_label=np.sort(output_label,axis=0)
output_label=pd.DataFrame(output_label,columns=['start','end'])
output_label.to_csv('Output-data/output_label(Data5).csv')  #记得改文件名
