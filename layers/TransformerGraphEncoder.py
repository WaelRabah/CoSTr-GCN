from abc import abstractmethod
from typing_extensions import Self

import torch
import torch.nn as nn   
from torch.autograd import Variable
from torch import Tensor
import math
import torch.nn.functional as F
import continual as co
from typing import  Tuple, Optional, Any, Union
from functools import partial
from continual import TensorPlaceholder


MaybeTensor=Union[Tensor, TensorPlaceholder]
State = Tuple[
    Tensor,  # Q_mem, (B, Nt-1, E)
    Tensor,  # K_T_mem, (B, E, Ns)
    Tensor,  # V_mem, (B, Ns, E)
]


def feed_forward(dim_input: int = 128, dim_feedforward: int = 512) -> nn.Module:
    return nn.Sequential(
        co.Linear(dim_input, dim_feedforward,dtype=torch.float),
        nn.Mish(),
        co.Linear(dim_feedforward, dim_input,dtype=torch.float),
    )
    
class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        
        self.norm = nn.LayerNorm(dimension,dtype=torch.float)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        x=self.dropout(self.sublayer(*tensors))
        # print(x.shape)
        # print(tensors[0].shape)
        x=tensors[0] + x
        x=self.norm(x)
        return x

def _scaled_dot_product_attention_default_state(
    batch_size: int,
    sequence_len: int,
    num_nodes : int,
    embed_dim_k: int,
    embed_dim_v: int,
    query_index=-1,
    init_fn=torch.zeros,
    dtype=None,
    device=None,
):
    init_fn = partial(init_fn, dtype=dtype, device=device)
    B = batch_size
    V=num_nodes
    N = sequence_len
    Nq = sequence_len - query_index - 1 if query_index >= 0 else -query_index - 1
    Q_mem = init_fn((B, V, Nq, embed_dim_k)).cuda()
    K_T_mem = init_fn((B, V, embed_dim_k, N)).cuda()
    V_mem = init_fn((B, V, N, embed_dim_v)).cuda()
    return (Q_mem, K_T_mem, V_mem)

def _clone_state(state):
    return [s.clone() for s in state]

def _scaled_dot_product_attention_step(
    prev_state: State,
    q_step: Tensor,  # step input (B, E)
    k_step: Tensor,  # step input (B, E)
    v_step: Tensor,  # step input (B, E)
    T,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, State]:
    """
    Computes the Continual Singe-output Scaled Dot-Product Attention on query, key and value tensors.
    Returns attended values and updated states.

    Args:
        q_step, k_step, v_step: query, key and value tensors for a step. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.

    Shape:
        - q_step: :math:`(B, V, E)` where B is batch size, V is the number of vertices and E is embedding dimension.
        - k_step: :math:`(B, V, E)` where B is batch size, V is the number of vertices and E is embedding dimension.
        - v_step: :math:`(B, V, E)` where B is batch size, V is the number of vertices and E is embedding dimension.

        - Output: attention values have shape :math:`(B, Nt, E)`; new state
    """
    # if attn_mask is not None:
    #     logger.warning("attn_mask is not supported yet and will be skipped")
    # if dropout_p != 0.0:
    #     logger.warning("dropout_p is not supported yet and will be skipped")

    (
        Q_mem,  # (B, V, Nq, E)
        K_T_mem,  # (B, V, E, Ns)
        V_mem,  # (B, V, Ns, E)
    ) = prev_state

    B, V, E = q_step.shape
    q_step = q_step / math.sqrt(E)
    q_sel = (Q_mem[:,:, 0] if Q_mem.shape[2] > 0 else q_step).unsqueeze(2)

    # Update states
    # Note: We're allowing the K and V mem to have one more entry than
    # strictly necessary to simplify computatations.

    K_T_new = torch.roll(K_T_mem, shifts=-1, dims=(3,))
    K_T_new[:, :, :, -1] = k_step

    V_new = torch.roll(V_mem, shifts=-1, dims=(2,))
    V_new[:, :, -1] = v_step
    
    attn = torch.bmm(q_sel.reshape(-1,1,E), K_T_new.reshape(-1,E,T))
    attn_sm = F.softmax(attn, dim=-1)
    
    
    if dropout_p > 0.0:
        attn_sm = F.dropout(attn_sm, p=dropout_p)
    
    # (B, V, Nt, Ns) x (B, V, Ns, E) -> (B, V, Nt, E)
    output = torch.bmm(attn_sm, V_new.reshape(-1,T,E)).reshape(B,V,-1,E)


    if Q_mem.shape[2] > 0:
        Q_new = torch.roll(Q_mem, shifts=-1, dims=(2,))
        Q_new[:, :, -1] = q_step
    else:
        Q_new = Q_mem

    new_states = (Q_new, K_T_new, V_new)
    return output, new_states


class AttentionHead( nn.Module, co.CoModule):
    def __init__(self, dim_in: int, dim_v: int, dim_k: int, kernel_size: int = 1 , stride :int =1, dropout :int=.1):
        super().__init__()
        self.call_mode = "forward_steps"
        self.embed_dim_second=False
        self.batch_first=True
        self.d_k=dim_k
        self.d_v=dim_v
        self.dropout=dropout
        self.q_conv=co.Conv2d(
                dim_in,
                dim_k,
                kernel_size=(kernel_size, 1),
                padding=(int((kernel_size - 1) / 2), 0),
                stride=(stride, 1),dtype=torch.float)
        self.k_conv=co.Conv2d(
                dim_in,
                dim_k,
                kernel_size=(kernel_size, 1),
                padding=(int((kernel_size - 1) / 2), 0),
                stride=(stride, 1),dtype=torch.float)
        self.v_conv=co.Conv2d(
                dim_in,
                dim_v,
                kernel_size=(kernel_size, 1),
                padding=(int((kernel_size - 1) / 2), 0),
                stride=(stride, 1),dtype=torch.float)

    def get_state(self) -> Optional[State]:
        """Get model state

        Returns:
            Optional[State]: A State tuple if the model has been initialised and otherwise None.
        """
        if (
            getattr(self, "Q_mem", None) is not None
            and getattr(self, "K_T_mem", None) is not None
            and getattr(self, "V_mem", None) is not None
            and getattr(self, "stride_index", None) is not None
        ):
            return (
                self.Q_mem,
                self.K_T_mem,
                self.V_mem,
                self.stride_index,
            )

    def set_state(self, state: State):
        """Set model state

        Args:
            state (State): State tuple to set as new internal internal state
        """
        (
            self.Q_mem,
            self.K_T_mem,
            self.V_mem,
            self.stride_index,
        ) = state

    def clean_state(self):
        """Clean model state"""
        if hasattr(self, "Q_mem"):
            del self.Q_mem
        if hasattr(self, "K_T_mem"):
            del self.K_T_mem
        if hasattr(self, "V_mem"):
            del self.V_mem
        if hasattr(self, "stride_index"):
            del self.stride_index

    def _forward_step(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        T: int,
        prev_state: State = None,
    ) -> Tuple[MaybeTensor, State]:
        """Forward computation for a single step with state initialisation

        Args:
            query, key, value: step inputs of shape `(B, E)` where B is the batch size and E is the embedding dimension.

        Returns:
            Tuple[MaybeTensor, State]: Step output and new state.
        """
        B, V, E = query.shape
        if prev_state is None:
            prev_state = (
                *_scaled_dot_product_attention_default_state(B, T, V, self.d_k, self.d_v),
                -T,
            )

        o, new_state = _scaled_dot_product_attention_step(
            prev_state[:-1],
            query,
            key,
            value,
            T,
            self.dropout,
        )
        stride_index = prev_state[-1]
        if stride_index < 0:
            stride_index += 1

        new_state = (*new_state, stride_index)
    
        return (
             o,
            new_state,
        )

    def forward_step(
        self,
        T: int,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        update_state=True,
    ) -> MaybeTensor:
        """
        Args:
            query, key, value: step_inputs for mapping a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.

        Shapes for inputs:
            - query: :math:`(N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.

        Shapes for outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        """
        if key is None:
            key = query
        if value is None:
            value = query

        tmp_state = self.get_state()

        if not update_state and tmp_state:
            backup_state = _clone_state(tmp_state)

        o, tmp_state = self._forward_step(query, key, value, T, tmp_state)
        if self.batch_first and not isinstance(o, TensorPlaceholder):
            o = o.transpose(1, 0)

        if update_state:
            self.set_state(tmp_state)
        elif tmp_state is not None:
            self.set_state(backup_state)

        return o

    def forward_steps(
        self,
        x : Tensor,
        update_state=True,
    ) -> MaybeTensor:
        """Forward computation for multiple steps with state initialisation

        Args:
            x (Tensor): input.
            update_state (bool): Whether internal state should be updated during this operation.

        Returns:
            Tensor: Stepwise layer outputs
        """
        B, T, V, E = x.shape
        query, key, value= self.projection(x)
        # if key is None:
        #     key = query
        # if value is None:
        #     value = query

        if self.embed_dim_second:
            # N E V T -> N T V E
            query = query.permute(0, 3, 2, 1)
            key = key.permute(0, 3, 2, 1)
            value = value.permute(0, 3, 2, 1)

        if self.batch_first:
            # N T V E -> T N V E
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tmp_state = self.get_state()

        if not update_state and tmp_state:
            backup_state = _clone_state(tmp_state)
        T = query.shape[0]
        assert T == key.shape[0]
        assert T == value.shape[0]
        outs = []
        for t in range(T):
            o, tmp_state = self._forward_step(query[t], key[t], value[t], T, tmp_state)
            if isinstance(o, Tensor):
                if self.batch_first:
                    o = o.transpose(0, 1)
                outs.append(o)
        # print("here",T,len(outs),outs[0].shape)
        if update_state:
            self.set_state(tmp_state)
        elif backup_state is not None:
            self.set_state(backup_state)

        if len(outs) == 0:
            return o

        o = torch.stack(outs, dim=2 ).squeeze(3).permute(1,2,0,3)

        return o

    def attention(self,Q,K,V):
      sqrt_dk=torch.sqrt(torch.tensor(self.d_k))
      attention_weights=F.softmax((Q @ K.transpose(-2,-1))/sqrt_dk)
      attention_vectors=attention_weights @ V
      return attention_vectors

    def projection(self,x: Tensor):
        x=x.permute(0,3,2,1)
        Q=self.q_conv(x).permute(0,3,2,1)
        K=self.k_conv(x).permute(0,3,2,1)
        V=self.v_conv(x).permute(0,3,2,1)
        return Q, K, V
      
    def forward(self, x: Tensor) -> Tensor:
    #   batch_size = x.size(0)
    #   seq_length = x.size(1)
    #   graph_size=x.size(2)

      # x=x.transpose(1,2)
      #Q, K, V=torch.split(self.qkv_conv(x), [self.d_k , self.d_k, self.d_v],
      #                            dim=1)





    #   x=self.attention(Q,K,V).transpose(1,2).contiguous().view(batch_size,seq_length,graph_size, self.d_k)
      return x

class MultiHeadAttention(co.CoModule,nn.Module):
    def __init__(self, num_heads: int, dim_in: int,dim_k,dim_q,dim_v,dropout):
        super().__init__()
        self.call_mode = "forward_steps"
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_v, dim_k,dropout=dropout) for _ in range(num_heads)]
        )
        self.linear = co.Linear(num_heads * dim_k, dim_in,dtype=torch.float)

    def forward_steps(self, x: Tensor, pad_end=False, update_state=True) -> Tensor:
        out=self.linear(
            torch.cat([h.forward_steps(x) for h in self.heads], dim=-1)
        )
        
        return out
    def forward(self, x) -> Tensor:
        return self.linear(
            torch.cat([h(x) for h in self.heads], dim=-1)
        )

class TransformerGraphEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 128,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_v=dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model,32,32,32,dropout),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(dim_model,dtype=torch.float)
    def forward(self, src: Tensor) -> Tensor:
        src = self.attention(self.norm(src))
        return self.feed_forward(src)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 200):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on z
        # pos and i
        pe = torch.zeros(max_seq_len,20 , d_model)
        for pos in range(max_seq_len):
          for node_id in range(0,20) :
            for i in range(0, d_model, 2):
                pe[pos, node_id, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, node_id, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        #self.learnable_pe=nn.Linear(d_model, d_model,dtype=torch.float)
        self.norm=nn.LayerNorm(d_model,dtype=torch.float)
        self.register_buffer('pe', pe)

    
    def forward(self, x):
        # make embeddings relatively larger
        # x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        
        x = self.norm(x + Variable(self.pe[:,:seq_len,:,:], \
        requires_grad=False).cuda())
        
        return x

class TransformerGraphEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 128,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
            TransformerGraphEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)      
            for _ in range(num_layers)
            ]
        )
        self.positional_encoder=PositionalEncoder(dim_model)
    def forward(self, x: Tensor) -> Tensor:
        x += self.positional_encoder(x)
        for layer in self.layers:
            x = layer(x)

        return x