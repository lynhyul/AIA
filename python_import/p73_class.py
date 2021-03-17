class Person:
    def __init__(self, name, age, addess) :
        self.name = name    # 나 자신의 이름은 ~~이다.
        self.age = age
        self.address = address

    def greeting(self) :    # 클래스안에서 생성되는 함수는 self를 넣어줘야 에러가 뜨지 않는다.
        print('안녕하세요, 저는 {0}입니다.'.format(self.name))
