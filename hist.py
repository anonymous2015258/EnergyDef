import numpy as np
import matplotlib.pyplot as plt

#sal1=np.load('orig_sal2_new21.npy')

sal1=np.load('orig_sal_ddnn_new1_8.npy')
sal1=sal1.flatten()

#sal1=np.sort(sal1)
print(sal1.sum())
val=np.percentile(sal1,100)*0.5

val2=np.percentile(sal1,100)
cnt=0
for v in sal1:
    if(v>val and v<val2):
        cnt+=1
print(cnt)
#print(np.where(sal1==np.percentile(sal1,50)))
#print(np.where(sal1==np.percentile(sal1,100)))
'''print(np.percentile(sal1,0))
print(np.percentile(sal1,25))
print(np.percentile(sal1,50))
print(np.percentile(sal1,75))


val=(np.percentile(sal1,75)-np.percentile(sal1,25))/(np.percentile(sal1,100)-np.percentile(sal1,0))
print(val)
print(sal1.sum())'''
plt.bar(np.arange(len(sal1)),sal1)

plt.show()

print('---------------')
#sal2=np.load('best_saliency2_new21.npy')
sal2=np.load('best_saliency_ddnn_new_7_8.npy')



sal2=sal2.flatten()
#sal2=np.sort(sal2)
print(sal2.sum())
val=np.percentile(sal2,100)*0.5
val2=np.percentile(sal2,100)
cnt=0
for v in sal2:
    if(v>val and v<val2):
        cnt+=1
print(cnt)
#print(np.where(sal2==np.percentile(sal2,50)))
#print(np.where(sal2==np.percentile(sal2,100)))
'''print(np.percentile(sal1,0))
print(np.percentile(sal1,25))
print(np.percentile(sal1,50))
print(np.percentile(sal1,75))
val=(np.percentile(sal1,75)-np.percentile(sal1,25))/(np.percentile(sal1,100)-np.percentile(sal1,0))
print(val)
print(sal1.sum())'''

plt.bar(np.arange(len(sal2)),sal2)

plt.show()

'''fig, ax = plt.subplots(1, 2)
ax[0].imshow(sal1, cmap='hot')
ax[0].axis('off')
ax[1].imshow(sal2, cmap='hot')
ax[1].axis('off')
plt.tight_layout()
fig.suptitle('The Image and Its Saliency Map')
plt.show()

sal2=sal2.flatten()
#sal2=np.sort(sal2)
print(sal2.sum())

sal1=sal1.flatten()
#sal2=np.sort(sal2)
print(sal1.sum())'''