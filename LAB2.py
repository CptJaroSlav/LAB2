#CWICZENIA
###4.1###

import numpy as np
import matplotlib.pyplot as plt

arr = np.array([1,2,3,4,5])
print(arr)

A = np.array([[1,2,3],[7,8,9]])
print(A)
A1 = np.array([[1, 2, 3],
[7, 8, 9]])
print(A1)
A2 = np.array([[1, 2, \
3],
[7, 8, 9]])
print(A2)

###4.1.2###
print('###4.1.2###')
v = np.arange(1,7)
print(v,"\n")
v1 = np.arange(-2,7)
print(v1,"\n")
v2 = np.arange(1,10,3)
print(v2,"\n")
v3 = np.arange(1,10.1,3)
print(v3,"\n")
v4 = np.arange(1,11,3)
print(v4,"\n")
v5 = np.arange(1,2,0.1)
print(v5,"\n")

###LINESPACE###

v6 = np.linspace(1,3,4)
print(v6)
v7 = np.linspace(1,10,4)
print(v7)

###POMOCNICZE###

X = np.ones((2,3))
Y = np.zeros((2,3,4))
Z = np.eye(2)
Q = np.random.rand(2,5) # np.round(10*np.random.rand((3,3)))
print(X,"\n\n",Y,"\n\n",Z,"\n\n",Q)

###4.1.3###

#U = np.block([[A],[X,Z]]) - mie można wykonać


###4.1.4###
print('###4.1.4###')
V = np.block([[
np.block([
np.block([[np.linspace(1,3,3)],
[np.zeros((2,3))]]) ,
np.ones((3,1))])
],
[np.array([100, 3, 1/2, 0.333])]] )
print(V)

###4.2###
print('###4.2###')
print( V[0,2] ) #tablice indeksowane od 0
print( V[3,0] )
print( V[3,3] )
print( V[-1,-1] )
print( V[-4,-3] )
print( V[3,:] )
print( V[:,2] )
print( V[3,0:3] )
print( V[np.ix_([0,2,3],[0,-1])] )
print( V[3] )

###4.3###
print('###4.3###')
Q = np.delete(V, 2, 0)
print(Q)
Q = np.delete(V, 2, 1)
print(Q)
v = np.arange(1,7)
print( np.delete(v, 3, 0) )

###4.4###
print('###4.4###')
np.size(v)
np.shape(v)
np.size(V)
np.shape(V)

###4.5###
print('###4.5###')
###4.5.1###
print('###4.5.1###')
A = np.array([[1, 0, 0],
[2, 3, -1],
[0, 7, 2]] )
B = np.array([[1, 2, 3],
[-1, 5, 2],
[2, 2, 2]] )
print( A+B )
print( A-B )
print( A+2 )
print( 2*A )

###4.5.2###
print('###4.5.2###')
MM1 = A@B
print(MM1)
MM2 = B@A
print(MM2)

###4.5.3###
print('###4.5.3###')
MT1 = A*B
print(MT1)
MT2 = B*A
print(MT2)

###4.5.4###
print('###4.5.4###')
DT1 = A/B
print(DT1)

###4.5.5###
print('###4.5.5###')
C = np.linalg.solve(A,MM1)
print(C) # porownaj z macierza B
x = np.ones((3,1))
b = A@x
y = np.linalg.solve(A,b)
print(y)

###4.5.6###
print('###4.5.6###')
PM = np.linalg.matrix_power(A,2) # por. A@A
PT = A**2 # por. A*A
print(PT)

###4.5.7###
print('###4.5.7###')
A.T # transpozycja
A.transpose()
A.conj().T # hermitowskie sprzezenie macierzy (dla m. zespolonych)
A.conj().transpose()
print(A)

###4.6###
print('###4.6###')
A == B
A != B
2 < A
A > B
A < B
A >= B
A <= B
np.logical_not(A)
np.logical_and(A, B)
np.logical_or(A, B)
np.logical_xor(A, B)
print( np.all(A) )
print( np.any(A) )
print( v > 4 )
print( np.logical_or(v>4, v<2))
print( np.nonzero(v>4) )
print( v[np.nonzero(v>4) ] )

###4.7###
print('###4.7###')
print(np.max(A))
print(np.min(A))
print(np.max(A,0))
print(np.max(A,1))
print( A.flatten() )
print( A.flatten('F') )

###5.1###
print('###5.1###')
x = [1,2,3]
y = [4,6,5]
plt.plot(x,y)
plt.show()

###5.1.1###
print('###5.1.1###')
x = np.arange(0.0, 2.0, 0.01)
y = np.sin(2.0*np.pi*x)
plt.plot(x,y)
plt.show()

###5.1.2###
print('###5.1.2###')
x = np.arange(0.0, 2.0, 0.01)
y = np.sin(2.0*np.pi*x)
plt.plot(x,y,'r:',linewidth=6)
plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Nasz pierwszy wykres')
plt.grid(True)
plt.show()

###5.1.3###
print('###5.1.3###')
x = np.arange(0.0, 2.0, 0.01)
y1 = np.sin(2.0*np.pi*x)
y2 = np.cos(2.0*np.pi*x)
plt.plot(x,y1,'r:',x,y2,'g')
plt.legend(('dane y1','dane y2'))
plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Wykres')
plt.grid(True)
plt.show()

###5.1.4###
print('###5.1.4###')
x = np.arange(0.0, 2.0, 0.01)
y1 = np.sin(2.0*np.pi*x)
y2 = np.cos(2.0*np.pi*x)
y = y1*y2
l1, = plt.plot(x,y,'b')
l2,l3 = plt.plot(x,y1,'r:',x,y2,'g')
plt.legend((l2,l3,l1),('dane y1','dane y2','y1*y2'))
plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Wykres')
plt.grid(True)
plt.show()

###3###
print('###3###')

A =np.block([[np.arange(1,6)], np.block([np.arange(6,1)])])
print(A)


