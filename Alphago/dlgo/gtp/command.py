class Command:
    def __init__(self, sequence, name, args):
        self.sequence = sequence
        self.name = name
        self.args = tuple(args)
    
    def __eq__(self, other):
        return self.sequence == other.sequence and \
            self.name == other.name and \
            self.args == other.args
    
    def __repr__(self):
        return 'Command(%r, %r, %r)' % (self.sequence, self.name, self.args)
    
    def __str__(self):
        return repr(self)


