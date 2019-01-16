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

Here I am trying to generate number of girl children and boy children in a family where the couple will keep having children till a boy is born
I have checked it for 10000 families and the ratio of boy to girl is near to 1 that means nearly equal number of boys and girls are born

"""

import numpy as np
ans=0
result = []
for child in range(10000):
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

    
        


