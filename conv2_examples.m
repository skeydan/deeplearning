pkg load signal

# "image"
A = [1,2,3;4,5,6;7,8,9]
# padded input matrix, for easier visualization
A_padded = [zeros(1,size(A,2)+2); [zeros(size(A,1),1), A, zeros(size(A,1),1)]; zeros(1,size(A,2)+2)]
# kernel
B = [1,0;0,0]

## real convolution
C_full = conv2(A,B,'full') # default
C_same = conv2(A,B,'same') 
C_valid = conv2(A,B,'valid')

## cross-correlation
XC = xcorr2(A,B) 