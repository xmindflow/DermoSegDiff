

class Singleton(type):
    _instance = None
    
    def __call__(self, *args, **kwargs):
        if self._instance == None:
            self._instance = super().__call__(*args, **kwargs)
        return self._instance
    

            