# Example from http://cs231n.github.io/convolutional-networks/#conv

import numpy as np

W = 6 # input width & height
D = 3 # input depth
F = 3 # filter size
K = 2 # number of filters
S = 2 # stride
P = 1 # padding
V = int((W - F + 2*P)/ S + 1) # output width & depth
# for stride=1, padding=1: V = W - 1 + 1 = W
print(V)


X_1 = [[0,0,0,0,0,0,0],
       [0,1,1,2,2,0,0],
       [0,0,0,0,1,0,0],
       [0,1,2,1,0,2,0],
       [0,2,2,1,1,0,0],
       [0,2,2,2,0,2,0],
       [0,0,0,0,0,0,0]]

X_2 = [[0,0,0,0,0,0,0],
       [0,0,2,0,1,2,0],
       [0,0,1,2,2,2,0],
       [0,0,1,2,0,0,0],
       [0,1,1,2,2,1,0],
       [0,2,2,1,0,1,0],
       [0,0,0,0,0,0,0]]

X_3 = [[0,0,0,0,0,0,0],
       [0,2,2,0,0,0,0],
       [0,0,0,1,1,2,0],
       [0,1,2,2,0,2,0],
       [0,1,1,1,2,2,0],
       [0,0,1,2,1,2,0],
       [0,0,0,0,0,0,0]]

X = np.array([X_1, X_2, X_3])
X = np.swapaxes(np.swapaxes(X,0,2),0,1)
print(X.shape)
#print(all(X_1 == X[:,:,0]) & all(X_2 == X[:,:,1]) & all(X_3 == X[:,:,2]))


W1_1 = [[-1,0,-1],
        [-1,0,-1],
        [-1,0,-1]]

W1_2 = [[-1,-1,1],
        [0,-1,0],
        [1,-1,1]]

W1_3 = [[0,0,0],
        [-1,1,1],
        [1,1,-1]]

W1 = np.array([W1_1, W1_2, W1_3])
W1 = np.swapaxes(np.swapaxes(W1,0,2),0,1)
#print(all(W1_1 == W1[:,:,0]) & all(W1_2 == X[:,:,1]) & all(W1_3 == X[:,:,2]))
print(W1.shape)

b1 = 0

C1 = np.empty([V,V,K])
# Second activation map
# elementwise multiplication, comprising all depth slices, then sum up
C1[0,0,1] = np.sum(X[:3,:3,:] * W1) + b1
C1[0,1,1] = np.sum(X[:3,2:5,:] * W1) + b1
C1[0,2,1] = np.sum(X[:3,4:7,:] * W1) + b1
C1[1,0,1] = np.sum(X[2:5,:3,:] * W1) + b1
C1[1,1,1] = np.sum(X[2:5,2:5,:] * W1) + b1
C1[1,2,1] = np.sum(X[2:5,4:7,:] * W1) + b1
C1[2,0,1] = np.sum(X[4:7,:3,:] * W1) + b1
C1[2,1,1] = np.sum(X[4:7,2:5,:] * W1) + b1
C1[2,2,1] = np.sum(X[4:7,4:7,:] * W1) + b1

C1_1 = C1[:,:,1]
print(C1_1)

'''
Implementation as matrix multiplication

1)
The local regions in the input image are stretched out into columns in an operation commonly called im2col. For example, if the input is [227x227x3] and it is to be convolved with 11x11x3 filters at stride 4, then we would take [11x11x3] blocks of pixels in the input and stretch each block into a column vector of size 11*11*3 = 363.
Iterating this process in the input at stride of 4 gives (227-11)/4+1 = 55 locations along both width and height, leading to an output matrix X_col of im2col of size [363 x 3025], where every column is a stretched out receptive field and there are 55*55 = 3025 of them in total.
Note that since the receptive fields overlap, every number in the input volume may be duplicated in multiple distinct columns.

2)
The weights of the CONV layer are similarly stretched out into rows. For example, if there are 96 filters of size [11x11x3] this would give a matrix W_row of size [96 x 363].

3)
The result of a convolution is now equivalent to performing one large matrix multiply np.dot(W_row, X_col), which evaluates the dot product between every filter and every receptive field location. In our example, the output of this operation would be [96 x 3025], giving the output of the dot product of each filter at each location.

4)
The result must finally be reshaped back to its proper output dimension [55x55x96].

This approach has the downside that it can use a lot of memory, since some values in the input volume are replicated multiple times in X_col. However, the benefit is that there are many very efficient implementations of Matrix Multiplication that we can take advantage of (for example, in the commonly used BLAS API). Moreover, the same im2col idea can be reused to perform the pooling operation, which we discuss next.
