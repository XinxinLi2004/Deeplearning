class Person:
    def __call__(self, name):
        print("__call__" + "hello" + name)

    def hello(self, name):
        print("hello" + name)

person = Person()###这里的类，没有init，不是构造方法，不能直接传入实参，__call__是魔法方法，故建立一个person函数，类似于功能函数，传入实参
person("wangshu")
person.hello("lisi")

class student:
    def __init__(self, name, age):
        self.name = name
        self.age = age






