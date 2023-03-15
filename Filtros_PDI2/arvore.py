class Node:
    def __init__(self, value, right, left, key=None):
        self.value = value
        self.right = right
        self.left  = left
        self.key   = key
        
    def __str__(self):
        return f'(Node[Value:{self.value},Key:{self.key}])'
        
    def percorrer(self,sequencia):
        retorno = self
        while len(sequencia)!=0:
            if   sequencia[0] == '0':
                retorno = retorno.left
            elif sequencia[0] == '1':
                retorno = retorno.right
            else:
                raise Exception(f'invalid sequence {sequencia}')
                
            sequencia = sequencia[1:]
                
        return retorno
    
    def isleaf(self):
        return self.left == None and self.right == None
    