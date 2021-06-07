import numpy as np
import matplotlib.pyplot as plt

#sal1=np.load('orig_sal2_new21.npy')

index_val=9
sal1=np.load('orig_sal_ddnn_etas1_3.npy')

#sal3=sal1[10:20]
#sal1=sal1[0:10]
#sal1=sal1.flatten()

#sal1=np.sort(sal1)

for s in sal1:
    s=s.flatten()
    print(s.sum())




'''print('#################')
s1=sal1[index_val].flatten()
s3=sal3[index_val].flatten()
val=np.percentile(s1,100)*0.5

val2=np.percentile(s1,100)
cnt=0
for v in s1:
    if(v>val and v<val2):
        cnt+=1
print(cnt)


val=np.percentile(s3,100)*0.5

val2=np.percentile(s3,100)
cnt=0
for v in s3:
    if(v>val and v<val2):
        cnt+=1
print(cnt)

print('#################')'''
#print(np.where(sal1==np.percentile(sal1,50)))
#print(np.where(sal1==np.percentile(sal1,100)))
'''print(np.percentile(sal1,0))
print(np.percentile(sal1,25))
print(np.percentile(sal1,50))
print(np.percentile(sal1,75))


val=(np.percentile(sal1,75)-np.percentile(sal1,25))/(np.percentile(sal1,100)-np.percentile(sal1,0))
print(val)
print(sal1.sum())'''
''''plt.bar(np.arange(len(sal1[8].flatten())),sal1[8].flatten())

plt.show()

print('---------------')

plt.bar(np.arange(len(sal3[index_val].flatten())),sal3[index_val].flatten())

plt.show()

print('---------------')'''
#sal2=np.load('best_saliency2_new21.npy')
sal2=np.load('best_saliency_ddnn_etas7_3.npy')

#print(sal2.shape)
#sal4=sal2[10:20]
#sal2=sal2[0:10]


#sal2=sal2.flatten()
#sal2=np.sort(sal2)

for s in sal2:
    s = s.flatten()
    print(s.sum())

'''for s in sal4:
    s = s.flatten()
    print(s.sum())'''
'''print(sal2.sum())
val=np.percentile(sal2,100)*0.5
val2=np.percentile(sal2,100)
cnt=0
for v in sal2:
    if(v>val and v<val2):
        cnt+=1
print(cnt)'''
#print(np.where(sal2==np.percentile(sal2,50)))
#print(np.where(sal2==np.percentile(sal2,100)))
'''print(np.percentile(sal1,0))
print(np.percentile(sal1,25))
print(np.percentile(sal1,50))
print(np.percentile(sal1,75))
val=(np.percentile(sal1,75)-np.percentile(sal1,25))/(np.percentile(sal1,100)-np.percentile(sal1,0))
print(val)
print(sal1.sum())'''

'''plt.bar(np.arange(len(sal2[8].flatten())),sal2[8].flatten())

plt.show()

plt.bar(np.arange(len(sal4[8].flatten())),sal4[8].flatten())

plt.show()

fig, ax = plt.subplots(1, 4)
ax[0].imshow(sal1[8], cmap='hot')
ax[0].axis('off')
ax[1].imshow(sal2[8], cmap='hot')
ax[1].axis('off')

ax[2].imshow(sal3[8], cmap='hot')
ax[2].axis('off')

ax[3].imshow(sal4[8], cmap='hot')
ax[3].axis('off')
plt.tight_layout()
fig.suptitle('The Image and Its Saliency Map')
plt.show()


print( np.argsort(-1*sal1[8].flatten())[:10])

print( np.argsort(-1*sal2[8].flatten())[:10])

print( np.argsort(-1*sal3[8].flatten())[:10])

print( np.argsort(-1*sal4[8].flatten())[:10])'''
'''sal2=sal2.flatten()
#sal2=np.sort(sal2)
print(sal2.sum())

sal1=sal1.flatten()
#sal2=np.sort(sal2)
print(sal1.sum())'''

print('#################')

'''s2=sal2[index_val].flatten()
s4=sal4[index_val].flatten()
val=np.percentile(s2,100)*0.5

val2=np.percentile(s2,100)
cnt=0
for v in s2:
    if(v>val and v<val2):
        cnt+=1
print(cnt)


val=np.percentile(s4,100)*0.5

val2=np.percentile(s4,100)
cnt=0
for v in s4:
    if(v>val and v<val2):
        cnt+=1
print(cnt)

print('#################')'''

fig, ax = plt.subplots(1, 10)
ax[0].imshow(sal1[0], cmap='hot')
ax[0].axis('off')
ax[1].imshow(sal2[0], cmap='hot')
ax[1].axis('off')

ax[2].imshow(sal1[1], cmap='hot')
ax[2].axis('off')

ax[3].imshow(sal2[1], cmap='hot')
ax[3].axis('off')

ax[4].imshow(sal1[2], cmap='hot')
ax[4].axis('off')

ax[5].imshow(sal2[2], cmap='hot')
ax[5].axis('off')

ax[6].imshow(sal1[3], cmap='hot')
ax[6].axis('off')

ax[7].imshow(sal2[3], cmap='hot')
ax[7].axis('off')

ax[8].imshow(sal1[4], cmap='hot')
ax[8].axis('off')

ax[9].imshow(sal2[4], cmap='hot')
ax[9].axis('off')


plt.tight_layout()
fig.suptitle('The Image and Its Saliency Map')
plt.show()

print( np.argsort(-1*sal1[0].flatten())[:10])

print( np.argsort(-1*sal1[1].flatten())[:10])
print( np.argsort(-1*sal1[2].flatten())[:10])
print( np.argsort(-1*sal1[3].flatten())[:10])

print( np.argsort(-1*sal1[4].flatten())[:10])

print('^^^^^^^^^^')

print( np.argsort(-1*sal2[0].flatten())[:10])

print( np.argsort(-1*sal2[1].flatten())[:10])
print( np.argsort(-1*sal2[2].flatten())[:10])
print( np.argsort(-1*sal2[3].flatten())[:10])

print( np.argsort(-1*sal2[4].flatten())[:10])

idxs=np.argsort(-1*sal1[0].flatten())[:10]

for i in range(1,5):
    print(sal1[i].flatten()[idxs[0]]," ",sal1[i].flatten()[idxs[1]]," ",sal1[i].flatten()[idxs[2]]," ",sal1[i].flatten()[idxs[3]]," ",sal1[i].flatten()[idxs[4]])

idxs=np.argsort(-1*sal2[0].flatten())[:10]

for i in range(1,5):
    print(sal2[i].flatten()[idxs[0]]," ",sal2[i].flatten()[idxs[1]]," ",sal2[i].flatten()[idxs[2]]," ",sal2[i].flatten()[idxs[3]]," ",sal2[i].flatten()[idxs[4]])

plt.bar(np.arange(len(sal1[2].flatten())),sal1[2].flatten())

plt.show()

plt.bar(np.arange(len(sal2[2].flatten())),sal2[2].flatten())

plt.show()

print(np.var(sal1[0]))
print(np.var(sal1[1]))
print(np.var(sal1[2]))
print(np.var(sal1[3]))
print(np.var(sal1[4]))

print('&&&&&&&')
print(np.var(sal2[0]))
print(np.var(sal2[1]))
print(np.var(sal2[2]))
print(np.var(sal2[3]))
print(np.var(sal2[4]))