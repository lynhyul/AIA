from machine.car import drive
from machine.tv import watch

drive() # 운전하다2
watch() # 시청하다2

print("================================")

from machine import car
from machine import tv

car.drive() # 운전하다2
tv.watch() # 시청하다2

print("================================")

from machine import car, tv

car.drive() # 운전하다2
tv.watch() # 시청하다2