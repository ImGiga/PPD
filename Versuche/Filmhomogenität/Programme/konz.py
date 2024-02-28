c_stock = 300 #mg/ml

# def amount(c_stock, c_target, v_target):
#     return (c_target * v_target) / c_stock

# v_target = 0.25 #ml

# c_target = [25, 50, 100, 150, 200, 250]
# sum = 0
# for c in c_target:
#     print("Benötigte Menge für " + str(c) + ": " + str(amount(c_stock, c, v_target)*1000) + "mu l")
#     print("Verbleibende Menge Chlor: " + str(v_target*1000 - amount(c_stock, c, v_target)*1000) + "mu l")
#     sum += amount(c_stock, c, v_target)*1000


# print("Benötigte Menge für" + str(1) + ": "+ str(amount(25, 1, 0.25)*1000) + "mu l")
# print("Verbleibende Menge Chlor: " + str(v_target*1000 - amount(25, 1, 0.25)*1000) + "mu l")
# sum += amount(25, 1, 0.25)*1000
# print("Summe: " + str(sum)+ "mu l")

def amount(c_stock, c_target, v_target):
    return (c_target * v_target) / c_stock

v_target = 0.25 #ml

c_target = [25, 50, 100, 150, 200, 250]
sum = 0

# Open the file in write mode
with open('concentration.txt', 'w') as f:
    for c in c_target:
        f.write("Benötigte Menge für " + str(c) + ": " + str(amount(c_stock, c, v_target)*1000) + "mu l\n")
        f.write("Verbleibende Menge Chlor: " + str(v_target*1000 - amount(c_stock, c, v_target)*1000) + "mu l\n")
        sum += amount(c_stock, c, v_target)*1000

    f.write("Benötigte Menge für" + str(1) + ": "+ str(amount(25, 1, 0.25)*1000) + "mu l\n")
    f.write("Verbleibende Menge Chlor: " + str(v_target*1000 - amount(25, 1, 0.25)*1000) + "mu l\n")
    sum += amount(25, 1, 0.25)*1000
    f.write("Summe: " + str(sum)+ "mu l\n")