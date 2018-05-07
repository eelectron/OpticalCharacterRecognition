#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 00:25:16 2018

@author: prashant
"""

#Print all hindi character
for i in range(2304, 2432):
    print(chr(i),end=' ')
    
    
print('\n')
'''
Print Prashant SIngh in Devanagiri script(HINDI)
'''
firstName = u'\u092A' +  u'\u094D'+ u'\u0930' + u'\u0936' + u'\u093E' +u'\u0902' +  u'\u0924'
lastName =   u'\u0938' +u'\u093F' +u'\u0902' u'\u0939'
print(firstName + ' '+ lastName)


#print unicode
name = 'प्रशांत'
print(type(name))  #type of object
s = name.encode('unicode-escape')
print(s)

a='त्व'
print(a.encode('unicode-escape'))

'''
Given unicode string how to convert it to letters ?
'''
eu = a.encode('unicode-escape')
du = eu.decode('unicode-escape')  #decode(...) method not work with String
print(type(eu))
print(du)

