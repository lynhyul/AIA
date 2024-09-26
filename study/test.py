
categories = ["0", "1", "2", "3","4","5","6","7",
                "8","9"] 
for idx, cat in enumerate(categories):
    label = [0 for i in range(10)]
    label[idx] = 1
    print(label)