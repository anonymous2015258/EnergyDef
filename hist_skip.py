import numpy as np
import matplotlib.pyplot as plt
from math import log,e
#sal1=np.load('orig_sal2_new21.npy')

index_val=9

def entropy(arr):
    ent=0
    arr=arr.flatten()
    for i in arr:
        #print(i)
        if(i!=0):
            ent-=i*log(i,2)
    return ent
sal1=np.load('orig_saliency_skip_gts_1_rl.npy')

#sal3=sal1[10:20]
#sal1=sal1[0:10]
#sal1=sal1.flatten()

#sal1=np.sort(sal1)

for s in sal1:
    s=s.flatten()
    #print(s.sum())
    print(entropy(s))

'''plt.bar(np.arange(len(sal1[48].flatten())),sal1[48].flatten())

plt.show()


plt.bar(np.arange(len(sal1[49].flatten())),sal1[49].flatten())

plt.show()'''
sal2=np.load('best_saliency_skip_gts_1_rl.npy')


print('---------------')
#sal3=sal1[10:20]
#sal1=sal1[0:10]
#sal1=sal1.flatten()

#sal1=np.sort(sal1)

for s in sal2:
    s=s.flatten()

    print(entropy(s))
    #print(s.sum())

'''plt.bar(np.arange(len(sal2[48].flatten())),sal2[48].flatten())

plt.show()

plt.bar(np.arange(len(sal2[49].flatten())),sal2[49].flatten())

plt.show()'''

print(sal1[0].flatten())
print(sal1[1].flatten())
print(sal2[0].flatten())
print(sal2[1].flatten())

fig, ax = plt.subplots(1, 4)
ax[0].imshow(sal1[3], cmap='hot')
ax[0].axis('off')
ax[1].imshow(sal1[2], cmap='hot')
ax[1].axis('off')

ax[2].imshow(sal2[3], cmap='hot')
ax[2].axis('off')
ax[3].imshow(sal2[2], cmap='hot')
ax[3].axis('off')
plt.tight_layout()
fig.suptitle('The Image and Its Saliency Map')
plt.show()