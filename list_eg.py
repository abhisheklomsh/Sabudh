mylist = input("Enter elements seperated bt space")
list = mylist.split()

for num in list:
	if int(num)>10:
		print(num)
	else:
		continue
