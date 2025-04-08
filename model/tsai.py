from .imports import *
from .layers import *
from .utils import *
from typing import Callable


class _SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)
        nn.init.normal_(self.spatial_proj.weight, std=1e-6)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u * v
        return out


class _gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = _SpatialGatingUnit(d_ffn, seq_len)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        return out


class _gMLPBackbone(nn.Module):
    def __init__(self, d_model=256, d_ffn=512, seq_len=256, depth=6):
        super().__init__()
        self.model = nn.Sequential(
            *[_gMLPBlock(d_model, d_ffn, seq_len) for _ in range(depth)]
        )

    def forward(self, x):
        return self.model(x)


class gMLP(_gMLPBackbone):
    def __init__(
        self,
        c_in,
        c_out,
        seq_len,
        patch_size=1,
        d_model=256,
        d_ffn=512,
        depth=6,
    ):
        assert seq_len % patch_size == 0, "`seq_len` must be divisibe by `patch_size`"
        super().__init__(d_model, d_ffn, seq_len // patch_size, depth)
        self.patcher = nn.Conv1d(
            c_in, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.head1 = nn.Linear(d_model, c_out)
        self.head2 = nn.Linear(d_model, c_out)

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_channels, _ = patches.shape
        patches = patches.permute(0, 2, 1)
        patches = patches.view(batch_size, -1, num_channels)
        embedding = self.model(patches)
        embedding = embedding.mean(dim=1)
        mlc = self.head1(embedding)
        reg = self.head2(embedding)
        return mlc, reg


class _TSSequencerEncoderLayer(nn.Module):
    def __init__(self, d_model:int, q_len:int=None, lstm_dropout:float=0., dropout:float=0, drop_path_rate:float=0.,
                 mlp_ratio:int=1, lstm_bias:bool=True, act:str='gelu', pre_norm:bool=False):
        super().__init__()
        self.bilstm = nn.LSTM(q_len, q_len, num_layers=1, bidirectional=True, bias=lstm_bias)
        self.dropout = nn.Dropout(lstm_dropout) if lstm_dropout else nn.Identity()
        self.fc = nn.Linear(2 * q_len, q_len)
        self.lstm_norm = nn.LayerNorm(d_model)
        self.pwff =  PositionwiseFeedForward(d_model, dropout=dropout, act=act, mlp_ratio=mlp_ratio)
        self.ff_norm = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate != 0 else nn.Identity()
        self.pre_norm = pre_norm
        self.transpose = Transpose(1,2)

    def forward(self, x):
        if self.pre_norm:
            x = self.drop_path(self.dropout(self.transpose(self.fc(self.bilstm(self.transpose(self.lstm_norm(x)))[0])))) + x
            x = self.drop_path(self.pwff(self.ff_norm(x))) + x
        else:
            x = self.lstm_norm(self.drop_path(self.dropout(self.transpose(self.fc(self.bilstm(self.transpose(x))[0])))) + x)
            x = self.ff_norm(self.drop_path(self.pwff(x)) + x)
        return x

# %% ../../nbs/069_models.TSSequencerPlus.ipynb 5
class _TSSequencerEncoder(nn.Module):
    def __init__(self, d_model, depth:int=6, q_len:int=None, lstm_dropout:float=0., dropout:float=0, drop_path_rate:float=0.,
                 mlp_ratio:int=1, lstm_bias:bool=True, act:str='gelu', pre_norm:bool=False):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        layers = []
        for i in range(depth):
            layer = _TSSequencerEncoderLayer(d_model, q_len=q_len, lstm_dropout=lstm_dropout, dropout=dropout, drop_path_rate=dpr[i],
                                      mlp_ratio=mlp_ratio, lstm_bias=lstm_bias, act=act, pre_norm=pre_norm)
            layers.append(layer)
        self.encoder = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()

    def forward(self, x):
        x = self.encoder(x)
        x = self.norm(x)
        return x

# %% ../../nbs/069_models.TSSequencerPlus.ipynb 6
class _TSSequencerBackbone(Module):
    def __init__(self, c_in:int, seq_len:int, depth:int=6, d_model:int=128, act:str='gelu',
                 lstm_bias:bool=True, lstm_dropout:float=0., dropout:float=0., drop_path_rate:float=0., mlp_ratio:int=1,
                 pre_norm:bool=False, use_token:bool=True,  use_pe:bool=True, n_cat_embeds:Optional[list]=None, cat_embed_dims:Optional[list]=None,
                 cat_padding_idxs:Optional[list]=None, cat_pos:Optional[list]=None, feature_extractor:Optional[Callable]=None,
                 token_size:int=None, tokenizer:Optional[Callable]=None):

        # Categorical embeddings
        if n_cat_embeds is not None:
            n_cat_embeds = listify(n_cat_embeds)
            if cat_embed_dims is None:
                cat_embed_dims = [emb_sz_rule(s) for s in n_cat_embeds]
            self.to_cat_embed = MultiEmbedding(c_in, n_cat_embeds, cat_embed_dims=cat_embed_dims, cat_padding_idxs=cat_padding_idxs, cat_pos=cat_pos)
            c_in, seq_len = output_size_calculator(self.to_cat_embed, c_in, seq_len)
        else:
            self.to_cat_embed = nn.Identity()

        # Sequence embedding
        if token_size is not None:
            self.tokenizer = SeqTokenizer(c_in, d_model, token_size)
            c_in, seq_len = output_size_calculator(self.tokenizer, c_in, seq_len)
        elif tokenizer is not None:
            if isinstance(tokenizer, nn.Module):  self.tokenizer = tokenizer
            else: self.tokenizer = tokenizer(c_in, d_model)
            c_in, seq_len = output_size_calculator(self.tokenizer, c_in, seq_len)
        else:
            self.tokenizer = nn.Identity()

        # Feature extractor
        if feature_extractor is not None:
            if isinstance(feature_extractor, nn.Module):  self.feature_extractor = feature_extractor
            else: self.feature_extractor = feature_extractor(c_in, d_model)
            c_in, seq_len = output_size_calculator(self.feature_extractor, c_in, seq_len)
        else:
            self.feature_extractor = nn.Identity()

        # Linear projection
        self.transpose = Transpose(1,2)
        if token_size is None and tokenizer is None and feature_extractor is None:
            self.linear_proj = nn.Linear(c_in, d_model)
            # self.linear_proj = nn.Conv1d(c_in, d_model, 1)
        else:
            self.linear_proj = nn.Identity()

        # Position embedding & token
        if use_pe:
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.use_pe = use_pe
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.use_token = use_token
        self.emb_dropout = nn.Dropout(dropout) if dropout else nn.Identity()

        # Encoder
        self.encoder = _TSSequencerEncoder(d_model, depth=depth, q_len=seq_len + use_token, lstm_bias=lstm_bias,
                                         lstm_dropout=lstm_dropout, dropout=dropout,
                                         mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate, act=act, pre_norm=pre_norm)

        self.flatten = nn.Flatten()
        self.head1 = nn.Linear(25600, 12)
        self.head2 = nn.Linear(25600, 12)

    def forward(self, x):

        # Categorical embeddings
        x = self.to_cat_embed(x)

        # Sequence embedding
        x = self.tokenizer(x)

        # Feature extractor
        x = self.feature_extractor(x)

        # Linear projection
        x = self.transpose(x)
        x = self.linear_proj(x)

        # Position embedding & token
        if self.use_pe:
            x = x + self.pos_embed
        if self.use_token: # token is concatenated after position embedding so that embedding can be learned using self.supervised learning
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.emb_dropout(x)

        # Encoder
        x = self.encoder(x)

        # Output
        x = x.transpose(1,2)
        x = self.flatten(x)

        mlc = self.head1(x)
        reg = self.head2(x)
        return mlc, reg

# %% ../../nbs/069_models.TSSequencerPlus.ipynb 7
class TSSequencerPlus(nn.Sequential):
    r"""Time Series Sequencer model based on:

    Tatsunami, Y., & Taki, M. (2022). Sequencer: Deep LSTM for Image Classification. arXiv preprint arXiv:2205.01972.
    Official implementation: https://github.com/okojoalg/sequencer

    Args:
        c_in:               the number of features (aka variables, dimensions, channels) in the time series dataset.
        c_out:              the number of target classes.
        seq_len:            number of time steps in the time series.
        d_model:            total dimension of the model (number of features created by the model).
        depth:              number of blocks in the encoder.
        act:                the activation function of positionwise feedforward layer.
        lstm_dropout:       dropout rate applied to the lstm sublayer.
        dropout:            dropout applied to to the embedded sequence steps after position embeddings have been added and
                            to the mlp sublayer in the encoder.
        drop_path_rate:     stochastic depth rate.
        mlp_ratio:          ratio of mlp hidden dim to embedding dim.
        lstm_bias:          determines whether bias is applied to the LSTM layer.
        pre_norm:           if True normalization will be applied as the first step in the sublayers. Defaults to False.
        use_token:          if True, the output will come from the transformed token. This is meant to be use in classification tasks.
        use_pe:             flag to indicate if positional embedding is used.
        n_cat_embeds:       list with the sizes of the dictionaries of embeddings (int).
        cat_embed_dims:     list with the sizes of each embedding vector (int).
        cat_padding_idxs:       If specified, the entries at cat_padding_idxs do not contribute to the gradient; therefore, the embedding vector at cat_padding_idxs
                            are not updated during training. Use 0 for those categorical embeddings that may have #na# values. Otherwise, leave them as None.
                            You can enter a combination for different embeddings (for example, [0, None, None]).
        cat_pos:            list with the position of the categorical variables in the input.
        token_size:         Size of the embedding function used to reduce the sequence length (similar to ViT's patch size)
        tokenizer:          nn.Module or callable that will be used to reduce the sequence length
        feature_extractor:  nn.Module or callable that will be used to preprocess the time series before
                            the embedding step. It is useful to extract features or resample the time series.
        flatten:            flag to indicate if the 3d logits will be flattened to 2d in the model's head if use_token is set to False.
                            If use_token is False and flatten is False, the model will apply a pooling layer.
        concat_pool:        if True the head begins with fastai's AdaptiveConcatPool2d if concat_pool=True; otherwise, it uses traditional average pooling.
        fc_dropout:         dropout applied to the final fully connected layer.
        use_bn:             flag that indicates if batchnorm will be applied to the head.
        bias_init:          values used to initialized the output layer.
        y_range:            range of possible y values (used in regression tasks).
        custom_head:        custom head that will be applied to the network. It must contain all kwargs (pass a partial function)
        verbose:            flag to control verbosity of the model.

    Input:
        x: bs (batch size) x nvars (aka features, variables, dimensions, channels) x seq_len (aka time steps)
    """

    def __init__(self, c_in:int, c_out:int, seq_len:int, d_model:int=128, depth:int=6, act:str='gelu',
                 lstm_dropout:float=0., dropout:float=0., drop_path_rate:float=0., mlp_ratio:int=1, lstm_bias:bool=True,
                 pre_norm:bool=False, use_token:bool=False, use_pe:bool=True,
                 cat_pos:Optional[list]=None, n_cat_embeds:Optional[list]=None, cat_embed_dims:Optional[list]=None, cat_padding_idxs:Optional[list]=None,
                 token_size:int=None, tokenizer:Optional[Callable]=None, feature_extractor:Optional[Callable]=None,
                 flatten:bool=False, concat_pool:bool=True, fc_dropout:float=0., use_bn:bool=False,
                 bias_init:Optional[Union[float, list]]=None, y_range:Optional[tuple]=None, custom_head:Optional[Callable]=None, verbose:bool=True,
                 **kwargs):
        if use_token and c_out == 1:
            use_token = False
            pv("use_token set to False as c_out == 1", verbose)
        backbone = _TSSequencerBackbone(c_in, seq_len, depth=depth, d_model=d_model, act=act,
                                      lstm_dropout=lstm_dropout, dropout=dropout, drop_path_rate=drop_path_rate,
                                      pre_norm=pre_norm, mlp_ratio=mlp_ratio, use_pe=use_pe, use_token=use_token,
                                      n_cat_embeds=n_cat_embeds, cat_embed_dims=cat_embed_dims, cat_padding_idxs=cat_padding_idxs, cat_pos=cat_pos,
                                      feature_extractor=feature_extractor, token_size=token_size, tokenizer=tokenizer)
        self.head_nf = d_model
        self.c_out = c_out
        self.seq_len = seq_len

        super().__init__(OrderedDict([('backbone', backbone)]))

######


class SampaddingConv1D_BN(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        self.padding = nn.ConstantPad1d((int((kernel_size - 1) / 2), int(kernel_size / 2)), 0)
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x):
        x = self.padding(x)
        x = self.conv1d(x)
        x = self.bn(x)
        return x


class build_layer_with_layer_parameter(Module):
    """
    formerly build_layer_with_layer_parameter
    """
    def __init__(self, layer_parameters):
        """
        layer_parameters format
            [in_channels, out_channels, kernel_size,
            in_channels, out_channels, kernel_size,
            ..., nlayers
            ]
        """
        self.conv_list = nn.ModuleList()

        for i in layer_parameters:
            # in_channels, out_channels, kernel_size
            conv = SampaddingConv1D_BN(i[0], i[1], i[2])
            self.conv_list.append(conv)

    def forward(self, x):

        conv_result_list = []
        for conv in self.conv_list:
            conv_result = conv(x)
            conv_result_list.append(conv_result)

        result = F.relu(torch.cat(tuple(conv_result_list), 1))
        return result


class OmniScaleCNN(Module):
    def __init__(self, c_in, c_out, seq_len, layers=[8 * 128, 5 * 128 * 256 + 2 * 256 * 128], few_shot=False):

        receptive_field_shape = seq_len//4
        layer_parameter_list = generate_layer_parameter_list(1,receptive_field_shape, layers, in_channel=c_in)
        self.few_shot = few_shot
        self.layer_parameter_list = layer_parameter_list
        self.layer_list = []
        for i in range(len(layer_parameter_list)):
            layer = build_layer_with_layer_parameter(layer_parameter_list[i])
            self.layer_list.append(layer)
        self.net = nn.Sequential(*self.layer_list)
        self.gap = GAP1d(1)
        out_put_channel_number = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_number = out_put_channel_number + final_layer_parameters[1]
        self.hidden1 = nn.Linear(out_put_channel_number, c_out)
        self.hidden2 = nn.Linear(out_put_channel_number, c_out)
    def forward(self, x):
        x = self.net(x)
        x = self.gap(x)
        # if not self.few_shot: x = self.hidden(x)
        mlc = self.hidden1(x)
        reg = self.hidden2(x)
        return mlc, reg

def get_Prime_number_in_a_range(start, end):
    Prime_list = []
    for val in range(start, end + 1):
        prime_or_not = True
        for n in range(2, val):
            if (val % n) == 0:
                prime_or_not = False
                break
        if prime_or_not:
            Prime_list.append(val)
    return Prime_list


def get_out_channel_number(paramenter_layer, in_channel, prime_list):
    out_channel_expect = max(1, int(paramenter_layer / (in_channel * sum(prime_list))))
    return out_channel_expect


def generate_layer_parameter_list(start, end, layers, in_channel=1):
    prime_list = get_Prime_number_in_a_range(start, end)

    layer_parameter_list = []
    for paramenter_number_of_layer in layers:
        out_channel = get_out_channel_number(paramenter_number_of_layer, in_channel, prime_list)

        tuples_in_layer = []
        for prime in prime_list:
            tuples_in_layer.append((in_channel, out_channel, prime))
        in_channel = len(prime_list) * out_channel

        layer_parameter_list.append(tuples_in_layer)

    tuples_in_layer_last = []
    first_out_channel = len(prime_list) * get_out_channel_number(layers[0], 1, prime_list)
    tuples_in_layer_last.append((in_channel, first_out_channel, 1))
    tuples_in_layer_last.append((in_channel, first_out_channel, 2))
    layer_parameter_list.append(tuples_in_layer_last)
    return layer_parameter_list

def gmlp(**kwargs):
    model = gMLP(c_in=2, c_out=12, seq_len=200).cuda()
    return model

def tss(**kwargs):
    model = TSSequencerPlus(c_in=2, c_out=12, seq_len=200).cuda()
    return model

def OmniScaleCNN(**kwargs):
    model = OmniScaleCNN(c_in=2, c_out=12, seq_len=200).cuda()
    return model

