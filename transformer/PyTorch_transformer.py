"""
A from scratch implementation of Transformer network

Original source code from:
    https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master

Raw implementation of Transformer architecture.    
"""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    SelfAttention.

    Args:
        embed_size: the embedding size.
        heads: the num_heads.
    """

    def __init__(
        self, embed_size, heads
    ):
        super(SelfAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(
            embed_size, embed_size
        )
        self.keys = nn.Linear(
            embed_size, embed_size
        )
        self.queries = nn.Linear(
            embed_size, embed_size
        )
        self.fc_out = nn.Linear(
            embed_size, embed_size
        )
    
    def forward(
        self,
        values,
        keys,
        query,
        mask,
    ):
        """
        Get number training examples.
        """
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)    # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)   # (N, query_len, embed_size)

        # Split the embedding into self.heads different pices.
        values = values.reshape(
            N, value_len, self.heads, self.head_dim
        )
        keys = keys.reshape(
            N, key_len, self.heads, self.head_dim
        )
        queries = queries.reshape(
            N, query_len, self.heads, self.head_dim
        )

        # Einsum does matrix mult. for query * Keys for each training example
        # with every other training example, don't be confused by eninsum
        # it's just how doing matrix multiplication & bmm

        energy = torch.einsum(
            "nqhd,nkhd->nhqk", [queries, keys]
        )
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indicies so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(
                mask == 0, float("-1e20")
            )
        
        # Normalize energy values similary to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability.
        attention = torch.softmax(
            energy / (self.embed_size ** (1 / 2)), dim=3
        )
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum(
            "nhql,nlhd->nqhd",
            [attention, values]
        ).reshape(N, query_len, self.heads * self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape : (N, value_len, headds, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim),
        # then reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
    """
    TransformerBlock.

    Args:
        embed_size: the embedding size.
        heads: the num_heads.
        dropout: the dropout.
        forward_expansion: the forward expansion.
    """

    def __init__(
        self,
        embed_size,
        heads,
        dropout,
        forward_expansion,
    ):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.fead_forward = nn.Sequential(
            nn.Linear(
                embed_size, forward_expansion * embed_size
            ),
            nn.ReLU(),
            nn.Linear(
                forward_expansion * embed_size, embed_size
            ),
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, value, key, query, mask
    ):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.fead_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out


class Encoder(nn.Module):
    """
    Encoder architecture.

    Args:
        src_vocab_size: the source vocab size.
        embed_size: the embedding size.
        num_layers: the num layers.
        heads: the num_heads.
        device: the device.
        forward_expansion: the forward expansion.
        dropout: the dropout.
        max_length: the max length.
    """

    def __init__(
        self,
        src_vocal_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(
            src_vocal_size, embed_size
        )
        self.position_embedding = nn.Embedding(
            max_length, embed_size
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, x, mask
    ):
        N, seq_length = x.shape
        positions = torch.arange(
            0, seq_length
        ).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, out)
        
        return out


class DecoderBlock(nn.Module):
    """
    DecoderBolock
    
    Args:
        embed_size: the embed size.
        heads: the num heads.
        forward_expansion: the forward dexpansion.
        dropout: the dropout.
        device: the device.
    """

    def __init__(
        self,
        embed_size,
        heads,
        forward_expansion,
        dropout,
        device,
    ):
        super(DecoderBlock, self).__init__()

        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(
            embed_size, heads=heads
        )
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        value,
        key,
        src_mask,
        trg_mask,
    ):
        attention = self.attention(
            x, x, x, trg_mask
        )
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(
            value, key, query, src_mask
        )

        return out


class Decoder(nn.Module):
    """
    Decoder architecture.

    Args:
        trg_vocab_size: the target vocab size.
        embed_size: the embed size.
        num_layers: the num layers.
        heads: the num heads.
        forward_expansion: the forward expansion.
        dropout: the dropout.
        device: the device.
        max_length: the max length.
    """

    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()

        self.device = device
        self.word_embedding = nn.Embedding(
            trg_vocab_size, embed_size
        )
        self.position_embedding = nn.Embedding(
            max_length, embed_size
        )

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size,
                    heads,
                    forward_expansion,
                    dropout,
                    device,
                )
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(
            embed_size, trg_vocab_size
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        enc_out,
        src_mask,
        trg_mask,
    ):
        N, seq_length = x.shape
        positions = torch.arange(
            0, seq_length
        ).expand(N, seq_length).to(self.device)
        x = self.dropout(
            (
                self.word_embedding(x) + \
                self.position_embedding(positions)
            )
        )

        for layer in self.layers:
            x = layer(
                x,
                enc_out,
                enc_out,
                src_mask,
                trg_mask,
            )
        
        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    """
    Desicion Transformer's hyperparameter.
    More information for hyperparameter:
        "Attention is all you need!"

    Args:
        src_vocab_size: the source vocab size.
        trg_vocab_size: the target vocab size.
        src_pad_idx: the soruce padding index.
        trg_pad_idx: the target padding index.
        embed_size: desicioned input/output size. also embedding vector's channel is
            d_model. each encoder and decoder keep this dimension when they transfer
            to next layer's encoder and decoder.
        num_layers: the transformer's encoder is 6 layers. And also decoder is 6 layers.
        forward_expansion:
        heads: the parallel partition num_heads
        dropout: the dropout
        deivce: use cpu or gpu?
        max_length: the max length.
    """

    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device='cpu',
        max_length=100,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        src_mask = (
            src != self.src_pad_idx
        ).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)

        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(
            torch.ones((trg_len, trg_len))
        ).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(
            trg, enc_src, src_mask, trg_mask
        )

        return out


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(device)

    x = torch.tensor([
        [1, 5, 6, 4, 3, 9, 5, 2, 0],
        [1, 8, 7, 3, 4, 5, 6, 7, 2],
    ]).to(device)

    trg = torch.tensor([
        [1, 7, 4, 3, 5, 9, 2, 0],
        [1, 5, 6, 2, 4, 7, 6, 2]
    ]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10

    model = Transformer(
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        device=device,
    ).to(device)

    out = model(x, trg[:, :-1])
    print(out.shape)