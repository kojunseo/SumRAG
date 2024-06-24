class AsIsProperty:
    def __init__(self, getter_func):
        self.getter_func = getter_func
        self.setter_func = None
    
    def __get__(self, obj, cls):
        return self.getter_func()