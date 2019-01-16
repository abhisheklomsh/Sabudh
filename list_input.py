"""
@author: abhisheklomsh

Here we are asking user to input numbers seperated by space and then we print numbers greater than limit provided by user
"""
mylist = input("Enter elements seperated by space")
list = mylist.split()
limit = input("Enter limit above which all numbers will be printed")
print("Here are numbers greater than "+limit)
for num in list:
	if int(num)>limit:
		print(num)
	else:
		continue
