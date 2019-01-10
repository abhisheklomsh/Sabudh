#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 09:57:01 2019

@author: abhisheklomsh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 11:48:12 2019

@author: abhisheklomsh
"""
import random
import numpy as np
ans=0
result = []
for child in range(100000):
    print(child)
    ans=0
    while(ans==0):
        print("you got ",ans)
        ans=np.random.randint(2)
        print("you got ",ans)
        result.append(ans)
        
        

girl=result.count(0)
boy=result.count(1)
print(result)
ratio= boy/girl
print(ratio)    

    
        


