import Model
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, batch_size, seq_length, input_size, embed_size, num_heads):
        super(Encoder, self).__init__()

        position = Model.Position(batch_size, seq_length, input_size)
        self.p = position.get_position()
        self.embedding = Model.Embedding(batch_size, seq_length, input_size)
        self.embedding_linear = nn.Linear(input_size, embed_size)
        
        self.multi_attention = Model.MultiAttention(embed_size, embed_size, num_heads)
        self.norm_attn = nn.LayerNorm(embed_size)

        self.feedforward = Model.FeedForward(embed_size, embed_size)
        self.norm_ff = nn.LayerNorm(embed_size)

    def forward(self, inputs, mask = False):
        inputs = inputs + self.p
        emb_x = self.embedding(inputs)
        emb_x = self.embedding_linear(emb_x)
        att_x = self.multi_attention(emb_x, mask)
        norm_x = self.norm_attn(att_x + emb_x)

        ff_x = self.feedforward(norm_x)
        x = self.norm_ff(ff_x + norm_x)

        return x

class Decoder(nn.Module):
    def __init__(self, batch_size, seq_de, input_de, seq_en, input_en, embed_size, num_heads):
        super(Decoder, self).__init__()
        
        # layers

        position = Model.Position(batch_size, seq_de, input_de)
        self.p = position.get_position()
        self.embedding = Model.Embedding(batch_size, seq_de, input_de)
        self.embedding_linear = nn.Linear(input_de, embed_size)

        self.multi_attention = Model.MultiAttention(embed_size, embed_size, num_heads)
        self.norm_attn = nn.LayerNorm(embed_size)

        self.feedforward = Model.FeedForward(embed_size, embed_size)
        self.norm_ff = nn.LayerNorm(embed_size)
        self.outputs = Model.Outputs(embed_size, input_de)

        # encoder的k，v; decoder的q
        self.encoder = Encoder(batch_size, seq_en, input_en, embed_size, num_heads)
        self.weight_en = Model.Weights(embed_size, embed_size, num_heads)
        self.weight_de = Model.Weights(embed_size, embed_size, num_heads)
        self.attention = Model.Attention(embed_size)
        

    def weight(self, inputs):
        Q, V, K = self.weight_en(inputs)
        return Q, V, K


    def forward(self, inputs_en, inputs_de):

        # get k, v
        outputs = self.encoder(inputs_en)
        # batch, seq_en, embed_size
        
        _, k, v = self.weight_en(outputs)
        # batch, heads, seq_en, hidden/heads

        # self decoder
        emb_x_de = self.embedding(inputs_de)
        emb_x_de = self.embedding_linear(emb_x_de)
        attn_x_de = self.multi_attention(emb_x_de, mask = True)
        x_de = self.norm_attn(attn_x_de + emb_x_de)
        # batch, seq_de, embedd

        # # get q: batch, heads, seq_de, hidden/heads
        q, _, _ = self.weight_de(x_de)
        # feed encoder
        att_o = self.attention(q, k, v)
        norm_o = self.norm_attn(att_o + x_de)
        ff_o = self.feedforward(norm_o)
        o = self.norm_ff(ff_o + norm_o)
        x = self.outputs(o)

        return x


# class ...
        # size
        # self.input_size = input_size
        # self.input_size = input_size
        
        # layers
        # ...


        # sequential
        # self.encoder_layer = nn.ModuleList([self.build_encoder_layer() for _ in range(num_layers)])
        # self.encoder_layer = self.build_encoder_layer()

    # def build_encoder_layer(self):
        # return nn.Sequential(
        #     self.embedding,
        #     # nn.LayerNorm(self.input_size),
        #     self.multi_attention,
        #     # nn.LayerNorm(self.hidden_size),
        #     self.feedforward,
        #     self.outputs
        # )

    # ...