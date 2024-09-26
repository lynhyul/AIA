import p11_car
import p12_tv

# p11_car.py의 module 이름은 : p11_car
# p12_tv.py의 module 이름은 : p12_car

print("==========================")
print("p13_do.py의 module 이름은 : ",__name__)
# p13_do.py의 module 이름은 : __main__
print("==========================")

p11_car.drive() # => 아무것도 안나옴
p12_tv.watch()  # => 아무것도 안나옴
