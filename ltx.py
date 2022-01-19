from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#归一化
def normalization(data):
    _range = np.max(abs(data))
    return data / _range

#滑动平滑
def move_avg(data,window_length,mode="same"):
    return(np.convolve(data, np.ones((window_length,))/window_length, mode=mode))


Data=pd.read_csv("meta_data/Data2.csv")

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
    x[...] = x * 1.4

Time=Time.flatten()

# plt.plot(Time,ans,label='max of sum_a and sum_w')
# #plt.plot(Time,sum_data_w,label='sum_a')
# plt.plot(Time,flip,label='value')
# plt.grid()
# plt.title("Processed data")
# plt.legend(ncol=1)
# plt.show()


gesture_point=np.array(np.where(ans>flip)).flatten()
transition_point=np.array(np.where(ans<=flip)).flatten()
print(gesture_point.shape)
g_point1=pd.Series(gesture_point)
t_point1=pd.Series(transition_point)
g_point1.to_csv('g_point1.csv')
t_point1.to_csv('t_point1.csv')
