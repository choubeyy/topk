class Vertex():
    def __init__(self, id, label):
        self.id = id
        self.label = label
        self.edges = []

class Edge():
    def __init__(self, frm, to, label, eprob) -> None:
        self.frm = frm
        self.to = to
        self.label = label
        self.eprob = eprob

class Graph():
    def __init__(self, id):
        self.id = id
        self.vertices = dict()
        self.edges = dict()
    
    def add_vertex(self, id, vlabel):
        self.vertices[id] = Vertex(id, vlabel)
    
    def add_edge(self, frm, to, elabel, eprob):
        self.vertices[frm].edges.append(Edge(frm, to, elabel, eprob))
        self.vertices[to].edges.append(Edge(to, frm, elabel, eprob))
        
        if to > frm:
            to, frm = frm, to
        self.edges[(to, frm)] = Edge(to, frm, elabel, eprob)




