import  pandas as pd
index = [1,2,3]
roll_values =[1,2,3]
Name = ["tom","hon","harray"]
Class = ["first","second","third"]
df = pd.DataFrame({"index":index,"roll":roll_values,"name":Name,"class":Class})
print(df)
df.to_csv("out.csv")
