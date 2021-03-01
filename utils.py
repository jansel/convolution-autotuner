class Once(set):
    def __call__(self, *x):
        return x not in self and (self.add(x) or True)