a = [1, 800]
b = [2, 3, 4, 5, 6, 700, ]

c = a + b
c.sort()
print(c)

a1=[]
a2=[]
for  each  in   range(len(c)):
    if  each %2==1:
        a1.append(c[each])
    else:
        a2.append(c[each])


print(a1,a2)


