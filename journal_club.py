import random
random.seed(0)

members = [
    "Martin", 
    "Jakob", 
    "Theo", 
    "Alex", 
    "Vaishnavi", 
    "Chandrashekar", 
    "Charlotte", 
    "Laya", 
    "Torben", 
    "Ilias", 
    "Baptiste", 
    "Giorgio", 
    "Lars", 
    "Gennaro", 
    "Matthew", 
    "Jakub", 
    "Dimitrii"
]


k = len(members)
print(k)
random.shuffle(members)
print(members)
