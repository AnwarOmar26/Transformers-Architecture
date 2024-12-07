import torch 
import torch.nn as nn


class SelfAttention(nn.Module): #nn.Module is a pytorch class that we will inherit
    def __init__(self, embed_size, heads): #head is nb of blocs ->> embed_size/heads in each bloc
        super(SelfAttention, self).__init__() #to inherit __init__ from Module
        self.embed_size =embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"  #if the nb isnt divisible it will give you this error
        
        #self.values = nn.Linear(self.head_dim, self.head_dim , bias = False) #create a linear relation of V = V.W -- input nb and output nb : head dimension
        #self.keys = nn.Linear(self.head_dim, self.head_dim , bias = False)
        #self.queries = nn.Linear(self.head_dim, self.head_dim , bias = False)
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)   #create a linear relation of V = V.W -- input nb and output nb : head dimension
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size) #the final output of similar shape to input
        #self.fc_out = nn.Linear (heads*self.head_dim , embed_size)   
    
    #PyTorch uses the forward method to define custom computations for a module. This allows flexibility in defining neural network layers or models that go beyond predefined operations.
    def forward(self , values , keys , query , mask):
        N= query.shape[0] #batch size
        value_len , key_len , query_len = values.shape[1] , keys.shape[1] , query.shape[1] #Sequence length

        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)
        #Split embedding into self.heads pieces
        values = values.reshape(N , value_len , self.heads , self.head_dim) # (batch size , sequence length , nb of attention head , dimension of each head )
        keys = keys.reshape(N , key_len , self.heads , self.head_dim)
        queries = queries.reshape(N , query_len , self.heads , self.head_dim)

        #einsum is matric multiplcation
        energy = torch.einsum ("nqhd,nkhd->nhqk" , [queries , keys])  # q:query len , h:head , d :head_dim , k : key len
        # queries shape : (N , query_len , heads , heads_dim)
        # keys shape : (N , key_len , heads , heads_dim)
        # energy shape : (N , heads , query_len , key_len)

        #Masking (Optional)
        if mask is not None : #if we send a mask we will shut down it off
            energy = energy.masked_fill(mask==0 , float("-1e20")) #If a mask is provided, certain positions in the energy tensor are set to -infinity to ignore them during the attention computation.
  
        #Scaling on energy
        energy_scaled = energy / (self.embed_size ** (1/2))
      
        #Do the Softmax to the scaled energy
        attention = torch.softmax(energy_scaled, dim = 3) #dim=3 to convert the energy scores into attention weights.

        #Attention * Values -> then we Concatunate
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim)  #Reshape here is for concatunate
        #attention shape : (N , heads , query_len , key_len)
        #values shape : (N , value_len , heads , heads_dim)
        #after einsum : (N , query_len , head , head_dim) then flatten last two dimension  -> Shape after reshaping: (N, query_len, embed_size) this is the size of input

        #Linear the Output
        out = self.fc_out(out)

        #return the result
        return out
    
class TransformerBlock(nn.Module) :
    def __init__ (self , embed_size , heads , dropout , forward_expansion):   #forward expansion to increase nb of output of first linear -- Ex of the NN if embed size = 64 and forward expansion = 8 : 64 - 512 - 64
        super (TransformerBlock , self).__init__()
        self.attention = SelfAttention(embed_size , heads) #create the multi-attention block
        self.norm1 = nn.LayerNorm(embed_size) #normalization (take an average)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size , forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size , embed_size)
        )
        self.dropout = nn.Dropout(dropout) #for prevent overfit

    def forward (self , value , key , query , mask):  #forward for use what we defined in the init 
      attention = self.attention(value , key , query , mask)  
      x =self.dropout(self.norm1(attention + query))
      forward = self.feed_forward(x)
      out = self.dropout(self.norm2(forward + x))
      return out
    
class Encoder (nn.Module) :
    def __init__(
        self ,
        src_vocab_size ,
        embed_size ,
        num_layers,
        heads ,
        device ,
        forward_expansion ,
        dropout , 
        max_length
    ) :
        super(Encoder , self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size , embed_size)
        self.position_embedding = nn.Embedding(max_length , embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size ,
                    heads ,
                    dropout = dropout ,
                    forward_expansion = forward_expansion
                )
            for _ in range (num_layers)]
        )
        self.dropout = nn.Dropout (dropout)

    def forward(self , x , mask) :
      N , seq_Length = x.shape
      positions = torch.arange(0 , seq_Length).to(self.device)

      out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

      for layer in self.layers :
           out = layer(out , out , out , mask)
      return out
    
class DecoderBlock(nn.Module) :
      def __init__ ( self , embed_size , heads , forward_expansion , dropout , device) :
          super(DecoderBlock , self).__init__()
          self.attention = SelfAttention(embed_size , heads)
          self.norm = nn.LayerNorm(embed_size)
          self.transformer_block = TransformerBlock (
              embed_size , heads , dropout , forward_expansion
          )
          self.dropout = nn.Dropout(dropout)

      def forward (self , x , value , key , src_mask , target_mask) :
          attention = self.attention (x,x,x, target_mask)
          query = self.dropout(self.norm(attention + x))
          out = self.transformer_block(value , key , query , src_mask)
          return out
      
class Decoder(nn.Module) :
      def __init__ (
          self ,
          trg_vocab_size , 
          embed_size ,
          num_layers ,
          heads , 
          forward_expansion ,
          dropout ,
          device ,
          max_length
      ) :
          super (Decoder , self).__init__()
          self.device = device 
          self.word_embedding = nn.Embedding(trg_vocab_size , embed_size)
          self.position_embedding = nn.Embedding(max_length , embed_size)

          self.Layers = nn.ModuleList (
              [DecoderBlock(embed_size , heads , forward_expansion , dropout , device)
              for _ in range (num_layers)]
          )

          self.fc_out = nn.Linear (embed_size , trg_vocab_size)
          self.dropout = nn.Dropout(dropout)

      def forward (self , x , enc_out , src_mask , trg_mask) : #enc : encoder
            N , seq_length = x.shape
            positions = torch.arange(0 , seq_length).to(self.device)
            x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

            for layer in self.Layers :
                x = layer(x , enc_out , enc_out , src_mask , trg_mask)
            
            out = self.fc_out(x)
            return out
      
class Transformers (nn.Module) :
      def __init__ (
          self ,
          src_vocab_size ,
          trg_vocab_size ,
          src_pad_idx ,
          trg_pad_idx ,
          embed_size = 256 ,
          num_layers = 6 ,
          forward_expansion = 4 ,
          heads = 8 ,
          dropout = 0 ,
          device = "cuda" ,
          max_length = 100
      ) :
            super(Transformers , self).__init__()
            self.encoder = Encoder(
                src_vocab_size ,
                embed_size ,
                num_layers ,
                heads ,
                device ,
                forward_expansion ,
                dropout ,
                max_length
            )

            self.decoder = Decoder(
                trg_vocab_size ,
                embed_size ,
                num_layers ,
                heads ,
                forward_expansion ,
                dropout ,
                device ,
                max_length
            )

            self.src_pad_idx = src_pad_idx
            self.trg_pad_idx = trg_pad_idx
            self.device = device

      def make_src_mask (self , src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

      def make_trg_mask (self , trg) :
        N , trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len , trg_len))).expand(
            N , 1 , trg_len , trg_len
        ) #triangulaire matrix
        return trg_mask.to(self.device)

      def forward (self , src , trg) :
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src , src_mask)
        out = self.decoder(trg , enc_src , src_mask , trg_mask)
        return out
      
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformers(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    out = model(x, trg[:, :-1])
    print(out.shape)