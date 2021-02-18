from machine.car import drive
from machine.tv import watch

drive() # 운전하다2
watch() # 시청하다2

print("================================")

# from machine import car
# from machine import tv
from machine import car, tv

car.drive() # 운전하다2
tv.watch() # 시청하다2

print("================================")

from machine.test.car import drive
from machine.test.tv import watch

drive() # test 운전하다2
watch() # test 시청하다2

print("================================")

from machine.test import car
from machine.test import tv

drive() # test 운전하다2
watch() # test 시청하다2

print("================================")

from machine import test

test.car.drive()
test.tv.watch()

# 환경변수 path를 통해서 비단 study 폴더 뿐만 아니라 다른 폴더에 있는것도 사용 할 수 있다.
