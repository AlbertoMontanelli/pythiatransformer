import logging
import argparse
import ROOT
import torch.nn as nn
import torch.optim as optimizer

from data_processing import inputs_tensor, outputs_tensor


'''
andrà fatta una funzione che PREPARA i dati che abbiamo simulato. per essere usati devono essere in questa forma: 
    source = particelle 23 che devono essere rappresentate come un tensore di torch e organizzati come (batch_size, seq_len, feature_dim).
        batch_size: numero di eventi/processi che stai trattando contemporaneamente.
        seq_len: numero di particelle in ciascun evento/processo (la lunghezza della sequenza).
        feature_dim: numero di caratteristiche per ciascuna particella (6 nel tuo caso: x, y, z, energia, massa, ID).
    
    target = stessa cosa ma con le particelle nello stato finale (penso serva per poi confrontarlo con la sequenza generata dal trasformeer)

    un esempio è: 
    particles_status_23 = torch.tensor([
        [
            [1.0, 2.0, 3.0, 100.0, 0.5, 1],   # Particella 1, evento 1
            [4.0, 5.0, 6.0, 200.0, 0.5, 2],   # Particella 2, evento 1
            [7.0, 8.0, 9.0, 300.0, 0.5, 3],   # Particella 3, evento 1
        ],
        [
            [2.0, 3.0, 4.0, 150.0, 0.6, 4],   # Particella 1, evento 2
            [5.0, 6.0, 7.0, 250.0, 0.6, 5],   # Particella 2, evento 2
            [8.0, 9.0, 10.0, 350.0, 0.6, 6],  # Particella 3, evento 2
        ]
    ], dtype=torch.float32)
    
    In questo caso: batch_size = 2 (due eventi), seq_len = 3 (tre particelle per evento), feature_dim = 6 (x, y, z, E, m, ID)
'''

'''
altre funzioni da fare:
1. allenamento del modello
2. validazione, test,...
altro?
'''

class Particle_Transformer(nn.Module):
    '''Transformer taking in input particles having status 23
    (i.e. outgoing particles of the hardest subprocess)
    and as target the final particles of the event.
    '''
    def __init__(self,
                 dim_features,
                 number_heads,
                 number_encoder_layers,
                 number_decoder_layers,
                 number_units,
                 dropout,
                 activation = nn.ReLU()
                ):
        '''
        Args: 
            dim_features (int): number of features of each particle
                                (px, py, pz, E, M, ID).
            number_heads (int): heads number of the attention system.
            number_encoder_layers (int): number of encoder layers.
            number_decoder_layers (int): number of decoder layers.
            number_units (int): number of units of each hidden layer.
            dropout (float): probability of each neuron to be 
                             switched off.
            activation (string): activation function of encoder 
                                 and/or decoder layers.
        '''

        super(Particle_Transformer, self).__init__()
        self.dim_features = dim_features
        self.number_heads = number_heads
        self.number_encoder_layers = number_encoder_layers
        self.number_decoder_layers = number_decoder_layers
        self.number_units = number_units 
        self.dropout = dropout
        self.activation = activation

    def data_preprocessor(self):
        '''
        Data preprocessing
        '''

        # Data normalization
        

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
            d_model = self.number_units, 
            # gli va passato il numero di features trasformato, non originali
            nhead = self.number_heads,
            number_encoder_layers = self.number_encoder_layers,
            number_decoder_layers = self.number_decoder_layers,
            dim_feedforward = 4 * self.number_units, 
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