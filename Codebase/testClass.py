def double(num):
    return num * 2

class Hello():
    def __init__(self):
        self.one = "Hello"
        self.two = "World"
        return

    def speak(self):
        print(self.one, " ", self.two)
        return


class echo():
    def __init__(self, word):
        self.word = word
        return

    def talk(self):
        print(self.word)
        return
