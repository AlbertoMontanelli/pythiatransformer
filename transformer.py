import logging
import argparse

import ROOT
import torch.nn as nn # questo contiene la classe transformer di pytorch
import torch.optim as optimizer # sicuramente ci servirà

'''
andrà fatta una funzione che PREPARA i dati che abbiamo simulato. per essere usati devono essere in questa forma: 
    source = particelle 23 che devono essere rappresentate come un tensore di torch e organizzati come (batch_size, seq_len, feature_dim).
        batch_size: numero di eventi/processi che stai trattando contemporaneamente.
        seq_len: numero di particelle in ciascun evento/processo (la lunghezza della sequenza).
        feature_dim: numero di caratteristiche per ciascuna particella (6 nel tuo caso: x, y, z, energia, massa, ID).
    
    target = stessa cosa ma con le particelle nello stato finale (penso serva per poi confrontarlo con la sequenza generata dal trasformeer)

    un esempio è: 
    particles_status_23 = torch.tensor([
        [1.0, 2.0, 3.0, 100.0, 0.5, 1],   # Particella 1, evento 1
        [4.0, 5.0, 6.0, 200.0, 0.5, 2],   # Particella 2, evento 1
        [7.0, 8.0, 9.0, 300.0, 0.5, 3],   # Particella 3, evento 1
        [2.0, 3.0, 4.0, 150.0, 0.6, 4],   # Particella 1, evento 2
        [5.0, 6.0, 7.0, 250.0, 0.6, 5],   # Particella 2, evento 2
        [8.0, 9.0, 10.0, 350.0, 0.6, 6],  # Particella 3, evento 2
    ], dtype=torch.float32)
    
    In questo caso: batch_size = 2 (due eventi), seq_len = 3 (tre particelle per evento), feature_dim = 6 (x, y, z, E, m, ID)
    Non riesco però a capire come fa il modello a sapere che la forma del tensore è questa
    LEO: eh infatti.... più che altro come fa a capire che i primi tre vettori stanno insieme e i secondi tre stanno insieme....
    questo è semplicemente l'elenco delle caratteristiche raggruppate per particella, ma poi le particelle non sono raggruppate 
    per evento.... vero è che se tu gli dici esplicitamente che il tensore è supponiamo (400, 3, 6) allora ok, ma questo vorrebbe dire
    che le 23 di ogni evento sono 3 (è il nostro caso?)
'''

'''
altre funzioni da fare:
1. allenamento del modello
2. validazione, test,...
altro?
'''

class Particle_Transformer(nn.Module): # va scritto con la maiuscola, o almeno, io ho trovato solo questo. poi se volevi module minuscolo ce lo dici te
    '''
        nn.module è una classe di torch per tutti i tuti di reti neurali, contiene delle subclassi come la classe transformer
        
        ma nn.module sarebbe un args della classe ??
    '''
    def __init__(self,
                 dim_features,
                 N_head,
                 N_encoder_layers,
                 N_decoder_layers,
                 N_units,
                 dropout,
                 activation = nn.ReLU()
                ):
        '''
        Args: 
            dim_features (int): number of features of the input (px, py, pz, E, M, ID, .. altro?)
            N_head (int): number of heads used for the attention mechanism
            N_encoder_layers (int): number of encoder layers
            N_decoder_layers (int): number of decoder layers
            N_units (int): number of units of each hidden layer
            dropout (float): probability of each neuron being switched off
            activation (string): activation function of encoder or/and decoder layers
        '''

        super(Particle_Transformer, self).__init__() 
        # serve per chiamare il costruttore della classe nn.Module
        # (è necessario) praticamente quello che vogliamo fare è 
        # estendere la classe di pytorch
        
        self.dim_features = dim_features
        self.N_head = N_head
        self.N_encoder_layers = N_encoder_layers
        self.N_decoder_layers = N_decoder_layers
        # N_units più è grande, più catturo informazioni dettagliate
        self.N_units = N_units 
        self.dropout = dropout
        self.activation = activation

    def data_preprocessor(self):
        '''
        Data preprocessing
        '''
        # queste righe servono per trasformare i dati di input e output
        # in una rappresentazione più adatta per un trasformer nn.Linear
        # è un layer con funzione lineare, applica la trasformazione 
        # y = xA^T + B
        self.input_projection = nn.Linear(self.dim_features, self.N_units) 
        # per lavorare con una rappresentazione più astratta. N_units > num_features
        self.output_projection = nn.Linear(self.N_units, self.dim_features) 
        # poi le riconverte alle dimensioni originali


    def transformer_application(self):
        '''

        '''
        self.transformer = nn.Transformer(
            d_model = self.N_units, 
            # gli va passato il numero di features trasformato, non originali
            nhead = self.N_head,
            num_encoder_layers = self.N_encoder_layers,
            num_decoder_layers = self.N_decoder_layers,
            dim_feedforward = 4 * self.N_units, 
            # non ho capito molto bene cos'è questo, non credo siano le
            # dimensioni delle hidden perché quello è il primo parametro
            # dim_feedforward rappresenta la dimensione degli strati 
            # completamente connessi all'interno del Transformer.
            #
            # È solitamente impostato a un valore maggiore rispetto a 
            # d_model (qui 4 * hidden_dim) per aumentare la capacità 
            # del modello di apprendere rappresentazioni più complesse
            # 4 * hidden_dim è una scelta comune, ma può essere 
            # modificato a seconda delle necessità specifiche del modello.
            dropout = self.dropout,
            activation = self.activation
            # il resto dei parametri li lascerei di default per ora
        )

    def forward(self, source, target):
        '''
        args:
            source (torch tensor: batch_size, seq_len, dim_feature): status 23 particles
            target (torch tensor: batch_size, seq_len, dim_feature): stable particles
        
        returns:
            output (torch tensor: batch_size, seq_len, dim_feature): 
        '''

        # qui sto praticamente usando il layer che ho costruito sopra
        # Proiezione nello spazio nascosto sia dei dati di input che
        # del target
        source = self.input_projection(source)  
        target = self.input_projection(target)  
        
        # qui sto usando il trasformer creato sopra
        output = self.transformer(source, target) 

        # anche qui uso l'altro layer che ho fatto prima
        # Proiezione finale nello spazio delle feature
        output = self.output_projection(output)
        return output