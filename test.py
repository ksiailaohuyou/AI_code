# from  collections  import Counter
# dict_1=Counter([1,2,3,3,2])
# import   copy
#
#
# sorted()


import   re
tels=['123456789','123456784','123456782']

for  tel in tels:
    ret=re.match('[\d]{8}[0-3,5-6,8-9]',tel)
    if  ret:
        print(ret.group())