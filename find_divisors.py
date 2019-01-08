num = int(input('Hi! Please enter the number to find divisors and press \'Enter\''))
mylist = []
for div in range(2,num):
	if num%div == 0:
		mylist.append(div)
	else:
		continue
print("Here are all divisors of "+str(num)+" "+str(mylist))