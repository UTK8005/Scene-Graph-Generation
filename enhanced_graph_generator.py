import os
import stanza
import re
from stanza.server.ud_enhancer import UniversalEnhancer
os.environ['CLASSPATH'] = ''
class NLP():
    def __init__(self):
        self.nlp=stanza.Pipeline("en")
    def build_output(self, sentence):
        doc = self.nlp(sentence)
        with UniversalEnhancer(language="en", classpath=os.environ['CLASSPATH']) as enhancer:
            result=enhancer.process(doc)
            tuples=[]
            token_tuple=()
            for token in result.sentence[0].token:
                token_tuple
                tuples.append((token.tokenEndIndex,token.word,token.lemma,token.pos,[]))
            for edge in result.sentence[0].enhancedDependencies.edge:
                for idx, tpl in enumerate(tuples):
                    if tpl[0] == edge.target:
                        tuples[idx][-1].append(edge.dep)
                        tuples[idx][-1].append(edge.source)
                        # Extract the word associated with the edge and add it to the tuple's list
                        word_associated_with_edge = [token.word for token in result.sentence[0].token if edge.source == token.tokenEndIndex]
                        tuples[idx][-1].extend(word_associated_with_edge)
            # Iterate through all tuples and split the last sublist if its length is greater than three
            # this has been done because in enhanced parsing a word can point to multiple source words for better understanding of relationships
            new_tuples = []
            for tpl in tuples:
                last_sublist = tpl[-1]
                if len(last_sublist) > 3:
                    middle_index = len(last_sublist) // 2
                    first_part = last_sublist[:middle_index]
                    second_part = last_sublist[middle_index:]
                    # Create a new tuple with the first part
                    new_tuples.append((tpl[0],tpl[1],tpl[2],tpl[3],first_part))
                    # Append a new tuple with the second part
                    new_tuples.append((tpl[0],tpl[1],tpl[2],tpl[3],second_part))
                else:
                    new_tuples.append(tpl)
        return new_tuples
class Graph():
    def __init__(self):
        self.entities = []
        self.actions = []
        self.attributes = []
        self.relations = []
        self.triplets = []
    def get_node(self,index):
        for entity in self.entities:
            if entity.index == index:
                return entity
        for action in self.actions:
            if action.index == index:
                return action
        for attribute in self.attributes:
            if attribute.index == index:
                return attribute
    def show(self):
        print ('Entities:')
        for entity in self.entities:
            print(entity.index, entity.word, entity.lemma, entity.pos)
        print ('\nActions:')
        for action in self.actions:
            print(action.index, action.word, action.lemma, action.pos)
        print ('\nAttributes:')
        for attribute in self.attributes:
            print(attribute.index, attribute.word, attribute.lemma, attribute.pos)
        print ('\nRelations:')
        for relation in self.relations:
            try:
                print(relation.node1.lemma, relation.relationship, relation.node2.lemma)
            except:
                print('Nonetype encountered')
        print("\n\n")
    def get_triplets(self, word_type = 'word'):
        if word_type == 'word':
            for relation in self.relations:
                try:
                    self.triplets.append(Triplet(relation.node1.word, relation.node2.word, relation.relationship))
                except:
                    continue
        if word_type == 'lemma':
            for relation in self.relations:
                try:
                    self.triplets.append(Triplet(relation.node1.lemma, relation.node2.lemma, relation.relationship))
                except:
                    continue
            for entity in self.entities:
                if entity.word.lower() != entity.lemma.lower():
                    self.triplets.append(Triplet(entity.lemma, entity.word, 'form'))
            for action in self.actions:
                if action.word.lower() != action.lemma.lower():
                    self.triplets.append(Triplet(action.lemma, action.word, 'form'))
            for attribute in self.attributes:
                if attribute.word.lower() != attribute.lemma.lower():
                    self.triplets.append(Triplet(attribute.lemma, attribute.word, 'form'))
        return(self.triplets)

class Entity():
    def __init__(self,index,word,lemma,pos):
        self.index = int(index)
        self.word = word
        self.lemma = lemma
        self.pos = pos
class Action():
    def __init__(self,index,word,lemma,pos):
        self.index = int(index)
        self.word = word
        self.lemma = lemma
        self.pos = pos
class Attribute():
    def __init__(self,index,word,lemma,pos):
        self.index = int(index)
        self.word = word
        self.lemma = lemma
        self.pos = pos
class Relation():
    def __init__(self,node1,node2,relation):
        self.node1 = node1
        self.node2 = node2
        self.relationship = relation
class Triplet():
    def __init__(self,word1,word2,relation):
        self.word1 = str(word1)
        self.word2 = str(word2)
        self.relation = str(relation)

import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(graph):
    G = nx.DiGraph()  # DiGraph means that the graph will be directed

    # Add entities, actions, and attributes as nodes
    for entity in graph.entities:
        G.add_node(entity.lemma, label=entity.lemma, type='entity')
    for action in graph.actions:
        G.add_node(action.lemma, label=action.lemma, type='action')
    for attribute in graph.attributes:
        G.add_node(attribute.lemma, label=attribute.lemma, type='attribute')

    # Add edges based on relations
    for relation in graph.relations:
        try:
            G.add_edge(relation.node1.lemma, relation.node2.lemma, label=relation.relationship)
        except AttributeError:
            pass
        

    # Draw the graph
    pos = nx.spring_layout(G)
    plt.figure(figsize=(5,5))

    nx.draw(G, pos, with_labels=True, node_size=300, node_color="skyblue", font_size=8, font_weight='bold', width=2.0, edge_color='gray')

    edge_labels = {(u, v): G[u][v]['label'] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

    plt.show()
    return G



def build_graph(nlp, sentences):
    graphlets=[]  # this correspond to a collection of them
    for sentence in sentences :
        sentence = re.sub(r'[^\w\s]', '', sentence) #"this is to keep sentence simplistic"
        tuples = nlp.build_output(sentence)
        graph = Graph() # this correspond to  single graphlets
        for tp in tuples:
            print(tp)
            index,word,lemma,pos,rels = tp
            if len(rels)!=0: # if you are not dealing with a root word
              if (pos.startswith('NN') or pos == 'PRP'):
                  entity = Entity(index,word,lemma,pos)
                  graph.entities.append(entity)
              if ((pos.startswith('VB') or pos.startswith('VBG')) and not rels[0].startswith('aux') and lemma !="be"):
                  action = Action(index,word,lemma,pos)
                  graph.actions.append(action)
              if (pos.startswith('JJ') or pos.startswith('RB') or pos.startswith('WRB')):
                  attribute = Attribute(index,word,lemma,pos)
                  graph.attributes.append(attribute)
            else:
              if ((pos.startswith('VB') or pos.startswith('VBG'))  and lemma !="be"):
                  action = Action(index,word,lemma,pos)
                  graph.actions.append(action)
              elif (pos.startswith('JJ') or pos.startswith('RB') or pos.startswith('WRB')):
                  attribute = Attribute(index,word,lemma,pos)
                  graph.attributes.append(attribute)
              elif (pos.startswith('NN') or pos == 'PRP' or pos == 'CD'):
                  entity = Entity(index,word,lemma,pos)
                  graph.entities.append(entity)

        for tp in tuples:
            relations = __get_relations(tp,tuples,graph)
            if relations is not None:
                for relation in relations:
                    if relation.relationship == 'cop':  # SPECIAL HANDLING OF COPULAR DEPENDENCIES (attribute assigned to subject)
                        for i,rel in enumerate(graph.relations):
                            if (rel.node2==relation.node2 and rel.relationship =='agent'):
                                if (rel.node2.pos.startswith('JJ') or rel.node2.pos.startswith('RB') or rel.node2.pos.startswith('CD') or rel.node2.pos.startswith('WRB')): # to handle cardinal numbers as well
                                    graph.relations[i].relationship = 'attr'
                                elif (rel.node2.pos.startswith('NN') or rel.node2.pos.startswith('PRP')):
                                    graph.relations[i].relationship = 'is'
                    else:
                        graph.relations.append(relation)
        print("\n")
        graph.show()
        graphlets.append(graph)
    return(graphlets)

def __get_relations(tp,tuples,graph):
      index,word,lemma,pos,rels  = tp
      relations = []
      if isinstance(rels,list) and len(rels)!=0: # this is because root word wont have any relation hence no rels
       dep,src_indx,src_name=rels
       if (dep in ['nsubj','nmod:agent', 'nsubj:xsubj']):  # AGENTS
           node1 = graph.get_node(int(index))
           if node1 == None:
               entity = Entity(index,word,lemma,pos)
               graph.entities.append(entity)
               node1 = graph.get_node(int(index))
           node2 = graph.get_node(int(src_indx))
           relationship= 'agent'
           relation = Relation(node1,node2,relationship)
           relations.append(relation)
       elif (dep in ['dobj','nsubj:pass','iobj','obj']):  # PATIENTS
           node1 = graph.get_node(int(index))
           if node1 == None:
               entity = Entity(index,word,lemma,pos)
               graph.entities.append(entity)
               node1 = graph.get_node(int(index))
           node2 = graph.get_node(int(src_indx))
           relationship= 'patient'
           relation = Relation(node2,node1,relationship)
           relations.append(relation)
       elif (dep.startswith('advcl') and ':' in dep):  # VERB MODIFIERS (PREPOSITION)
           node1 = graph.get_node(int(src_indx))
           node2 = graph.get_node(int(index))
           if node2 == None:
               entity = Entity(index,word,lemma,pos)
               graph.entities.append(entity)
               node2 = graph.get_node(int(index))
           relationship=dep.split(':')[1]
           relation = Relation(node1,node2,relationship)
           relations.append(relation)
       elif (dep.startswith('nmod:')):    # NOUN MODIFIERS (PREPOSITIONS)
           node1 = graph.get_node(int(src_indx))
           node2 = graph.get_node(int(index))
           if node2 == None:
               entity = Entity(index,word,lemma,pos)
               graph.entities.append(entity)
               node2 = graph.get_node(int(index))
           relationship=dep.split(':')[1]
           if relationship == 'poss':      # POSSESSIVE PRONOUNS ARE DENOTED BY 'OF' RELATIONS
               relationship = 'of'
           relation = Relation(node1,node2,relationship)
           relations.append(relation)
       elif (dep in ('amod','advmod','neg','nummod')):  # NOUN/VERB MODIFIERS (ATTRIBUTES)
           node1 = graph.get_node(int(src_indx))
           node2 = graph.get_node(int(index))
           if node2 == None:
               entity = Entity(index,word,lemma,pos)
               graph.entities.append(entity)
               node2 = graph.get_node(int(index))
           relationship='attr'
           relation = Relation(node1,node2,relationship)
           relations.append(relation)
       elif (dep == 'cop'):                            # MARKED FOR SPECIAL HANDLING (LINE 133)
           node1 = graph.get_node(int(index))
           node2 = graph.get_node(int(src_indx))
           relationship='cop'
           relation = Relation(node1,node2,relationship)
           relations.append(relation)
       elif (dep == 'xcomp'):                          # CLAUSES (NEEDS TO BE CONNECTED WITH APPROPRIATE PREPOSITIONS)
           node1 = graph.get_node(int(src_indx))
           node2 = graph.get_node(int(index))
           flag = 0
           for tp1 in tuples:
               index1,word1,lemma1,pos1,rels1 = tp1
               for rel1 in rels1:
                   if rel1[0]=='mark' and rel1[1] == index and lemma1 == 'to':
                       relationship = 'to'
                       flag = 1
                   if rel1[0]=='nsubj' and rel1[1] == index :
                       relationship = 'is'
                       node1 = graph.get_node(index1)
                       flag = 1
           if flag == 0:
               for rel in graph.relations:
                   if rel.node2 == node1:
                       node1 = rel.node1
                       relationship = 'attr'          # CLAUSES NOT CONNECTED BY PREPOSITIONS BECOME ATTRIBUTES
           relation = Relation(node1,node2,relationship)
           relations.append(relation)
       elif (dep == 'appos'):                         #
           node1 = graph.get_node(int(src_indx))
           node2 = graph.get_node(int(index))
           relationship='is'
           relation = Relation(node1,node2,relationship)
           relations.append(relation)
       elif (dep=='case'):
           node1 = graph.get_node(int(src_indx))
           node2 = graph.get_node(tuples[int(src_indx)-1][0])  # this is for handling oblique dependencies
           relationship= lemma
           relation = Relation(node2,node1,relationship)
           relations.append(relation)
       elif (dep in ['aux','auxpass']):
           node1 = graph.get_node(int(src_indx))
           node2 = graph.get_node(int(index))
           if node2 == None:
               attr = Attribute(index,word,lemma,pos)
               graph.attributes.append(attr)
               node2 = graph.get_node(int(index))
           relationship='aux'
           relation = Relation(node1,node2,relationship)
           relations.append(relation)
       elif (dep.startswith('conj:')):
           node1 = graph.get_node(int(src_indx))
           node2 = graph.get_node(int(index))
           relationship=dep.split(':')[1]
           relation = Relation(node1,node2,relationship)
           relations.append(relation)
       elif (dep.startswith('conj')):
           node1 = graph.get_node(int(src_indx))
           node2 = graph.get_node(int(index))
           relationship='conjunction'
           relation = Relation(node1,node2,relationship)
           relations.append(relation)
       elif (dep in ['ccomp','csubj']):
           node1 = graph.get_node(int(index))
           node2 = graph.get_node(int(src_indx))
           relationship= 'that'
           relation = Relation(node1,node2,relationship)
           relations.append(relation)
       elif (dep == 'dep'):
           node1 = graph.get_node(int(index))
           node2 = graph.get_node(int(src_indx))
           relationship= 'dep'
           relation = Relation(node1,node2,relationship)
           relations.append(relation)
       elif (dep=='compound'):
           node1 = graph.get_node(int(index))
           node2 = graph.get_node(int(src_indx))
           relationship= 'compound'
           relation = Relation(node1,node2,relationship)
           relations.append(relation)

       else:
           relation = None
       return (relations)
