import numpy as np
import matplotlib.pyplot as plt



# moving average
alpha = 0.02


data1 = np.loadtxt('../dtc_frame_skip4/experiments/VizDoom/performance.txt')
print(data1.shape)
mva1 = np.zeros((data1.shape[0]),dtype=np.float)
for i in range(data1.shape[0]):
   if i == 0:
     mva1[i] = data1[i,2]
   else:
     mva1[i] = alpha * data1[i,2] + (1.0 - alpha) * mva1[i-1] 


data2 = np.loadtxt('../dtc_frame_skip8/experiments/VizDoom/performance.txt')
print(data2.shape)
mva2 = np.zeros((data2.shape[0]),dtype=np.float)
for i in range(data2.shape[0]):
   if i == 0:
     mva2[i] = data2[i,2]
   else:
     mva2[i] = alpha * data2[i,2] + (1.0 - alpha) * mva2[i-1] 




plt.plot(data1[:,0], data1[:,2])
plt.plot(data1[:,0], mva1, label='frame skip = 4')
plt.xlabel('episode #', fontsize=15)
plt.xlim(0, 12000)
plt.ylabel('episode reward', fontsize=15)
plt.legend(fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('vizdoom_dtc_fs4.png')
plt.clf()
plt.close()


plt.plot(data2[:,0], data2[:,2])
plt.plot(data2[:,0], mva2, label='frame skip = 8')
plt.xlabel('episode #', fontsize=15)
plt.xlim(0, 12000)
plt.ylabel('episode reward', fontsize=15)
plt.legend(fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('vizdoom_dtc_fs8.png')
plt.clf()
plt.close()


#-----------------------------------------------------------------


data1 = np.loadtxt('../basic2_skip4/experiments/VizDoom/performance.txt')
print(data1.shape)
mva1 = np.zeros((data1.shape[0]),dtype=np.float)
for i in range(data1.shape[0]):
   if i == 0:
     mva1[i] = data1[i,2]
   else:
     mva1[i] = alpha * data1[i,2] + (1.0 - alpha) * mva1[i-1] 


data2 = np.loadtxt('../basic2_skip8/experiments/VizDoom/performance.txt')
print(data2.shape)
mva2 = np.zeros((data2.shape[0]),dtype=np.float)
for i in range(data2.shape[0]):
   if i == 0:
     mva2[i] = data2[i,2]
   else:
     mva2[i] = alpha * data2[i,2] + (1.0 - alpha) * mva2[i-1] 


data3 = np.loadtxt('../basic2_skip10/experiments/VizDoom/performance.txt')
print(data3.shape)
mva3 = np.zeros((data3.shape[0]),dtype=np.float)
for i in range(data3.shape[0]):
   if i == 0:
     mva3[i] = data3[i,2]
   else:
     mva3[i] = alpha * data3[i,2] + (1.0 - alpha) * mva3[i-1] 




plt.plot(data1[:,0], data1[:,2])
plt.plot(data1[:,0], mva1, label='frame skip = 4')
plt.xlabel('episode #', fontsize=15)
plt.xlim(0, 8000)
plt.ylabel('episode reward', fontsize=15)
plt.legend(fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('vizdoom_basic_fs4.png')
plt.clf()
plt.close()


plt.plot(data2[:,0], data2[:,2])
plt.plot(data2[:,0], mva2, label='frame skip = 8')
plt.xlabel('episode #', fontsize=15)
plt.xlim(0, 8000)
plt.ylabel('episode reward', fontsize=15)
plt.legend(fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('vizdoom_basic_fs8.png')
plt.clf()
plt.close()


plt.plot(data3[:,0], data3[:,2])
plt.plot(data3[:,0], mva3, label='frame skip = 10')
plt.xlabel('episode #', fontsize=15)
plt.xlim(0, 8000)
plt.ylabel('episode reward', fontsize=15)
plt.legend(fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('vizdoom_basic_fs10.png')
plt.clf()
plt.close()



