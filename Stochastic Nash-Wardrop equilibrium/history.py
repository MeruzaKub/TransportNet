class History():
    """
    history handler
    """
    def __init__(self, *attributes):
        self.dict = {}
        self.attributes = list(attributes)
        for attribute in self.attributes:
            self.dict[attribute] = []
        
    def update(self, *values):
        for index, value in enumerate(values):
            self.dict[self.attributes[index]].append(value)
            
    def get(self, attribute):
        return self.dict[attribute]