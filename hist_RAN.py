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
sal1=np.load('orig_saliency_RAN_acc_20_2.npy')

#sal3=sal1[10:20]
#sal1=sal1[0:10]
#sal1=sal1.flatten()

#sal1=np.sort(sal1)

'''for s in sal1:
    s=s.flatten()
    print(s.sum())

    #print(s)
    norm=np.linalg.norm(s)
    s=s/norm

    #print(s)
    print(entropy(s))
    print(np.var(s))'''


for sl in sal1:
    for ind in range(10):
        s=sl[ind].flatten()
        print(s.sum())

    #print(s)
        #norm=np.linalg.norm(s)
        #s=s/norm

    #print(s)
        print(entropy(s))
        print(np.var(s))


#plt.bar(np.arange(len(sal1[0].flatten())),sal1[0].flatten())

#plt.show()

#plt.bar(np.arange(len(sal1[1].flatten())),sal1[1].flatten())

#plt.show()



sal2=np.load('best_saliency_RAN_acc_20_8.npy')


print('---------------')
#sal3=sal1[10:20]
#sal1=sal1[0:10]
#sal1=sal1.flatten()

#sal1=np.sort(sal1)

'''for s in sal2:
    s=s.flatten()
    print(s.sum())
    norm = np.linalg.norm(s)
    s = s / norm
    print(entropy(s))
    print(np.var(s))'''


for sl in sal2:
    for ind in range(10):
        s=sl[ind].flatten()
        print(s.sum())


    #print(s)
        #norm=np.linalg.norm(s)
        #s=s/norm

    #print(s)
        print(entropy(s))
        print(np.var(s))
    print('#######')


'''plt.bar(np.arange(len(sal2[0].flatten())),sal2[0].flatten())

plt.show()
plt.bar(np.arange(len(sal2[1].flatten())),sal2[1].flatten())

plt.show()

fig, ax = plt.subplots(1, 7)
ax[0].imshow(sal2[0], cmap='hot')
ax[0].axis('off')
ax[1].imshow(sal2[1], cmap='hot')
ax[1].axis('off')

ax[2].imshow(sal2[2], cmap='hot')
ax[2].axis('off')
ax[3].imshow(sal2[3], cmap='hot')
ax[3].axis('off')


ax[4].imshow(sal1[0], cmap='hot')
ax[4].axis('off')
ax[5].imshow(sal1[1], cmap='hot')
ax[5].axis('off')

ax[6].imshow(sal1[2], cmap='hot')
ax[6].axis('off')
#ax[3].imshow(sal1[3], cmap='hot')
#ax[3].axis('off')
plt.tight_layout()
fig.suptitle('The Image and Its Saliency Map')
plt.show()
print('######')

print((np.square(sal2[0].flatten()-sal2[1].flatten())).mean())
print('######')

#print((np.square(sal2[1].flatten()-sal2[2].flatten())).mean())

print((np.square(sal2[3].flatten()-sal2[4].flatten())).mean())


print((np.square(sal2[3].flatten()-sal2[7].flatten())).mean())

print((np.square(sal2[4].flatten()-sal2[7].flatten())).mean())

print((np.square(sal2[5].flatten()-sal2[7].flatten())).mean())
print('######')

print((np.square(sal1[0].flatten()-sal1[1].flatten())).mean())
print((np.square(sal1[1].flatten()-sal1[2].flatten())).mean())'''
'''print((np.square(sal1[3].flatten()-sal1[4].flatten())).mean())


print((np.square(sal1[3].flatten()-sal1[7].flatten())).mean())

print((np.square(sal1[4].flatten()-sal1[7].flatten())).mean())

print((np.square(sal1[5].flatten()-sal1[7].flatten())).mean())'''
'''print('######')
print((np.square(sal1[2]-sal1[3])).mean())
print((np.square(sal2[2]-sal2[3])).mean())

print('######')
print((np.square(sal1[3]-sal1[4])).mean())
print((np.square(sal2[6]-sal2[7])).mean())'''

fig, ax = plt.subplots(1, 7)
ax[0].imshow(sal2[0][0], cmap='hot')
ax[0].axis('off')
ax[1].imshow(sal2[0][1], cmap='hot')
ax[1].axis('off')

ax[2].imshow(sal2[0][2], cmap='hot')
ax[2].axis('off')
ax[3].imshow(sal2[0][3], cmap='hot')
ax[3].axis('off')


ax[4].imshow(sal1[0][0], cmap='hot')
ax[4].axis('off')
ax[5].imshow(sal1[0][1], cmap='hot')
ax[5].axis('off')

ax[6].imshow(sal1[0][2], cmap='hot')
ax[6].axis('off')
#ax[3].imshow(sal1[3], cmap='hot')
#ax[3].axis('off')
plt.tight_layout()
fig.suptitle('The Image and Its Saliency Map')
plt.show()