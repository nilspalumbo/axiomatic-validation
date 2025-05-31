# Adapted from https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/main/Grokking_Analysis.ipynb

# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
import tqdm.notebook as tqdm

import random
import time

# from google.colab import drive
from pathlib import Path
import pickle
import os

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "vscode"

import plotly.graph_objects as go

from torch.utils.data import DataLoader

from functools import *
import pandas as pd
import gc

# import comet_ml
import itertools

from plot import *
from model import (
    HookPoint,
    Embed,
    Unembed,
    PosEmbed,
    LayerNorm,
    Attention,
    MLP,
    TransformerBlock,
)

import os
# TODO: adapt for final structure
high_level_root = Path(os.getcwd())
saved_runs_dir = high_level_root/'saved_runs'
large_file_root = high_level_root/'data'

try:
    os.mkdir(large_file_root)
except:
    pass

saved_run_name = "grokking_addition_full_run.pth"
if saved_run_name not in os.listdir(large_file_root):
    print('Downloading saved model from "Progress measure for grokking..."')
    os.system(f"gdown 12pmgxpTHLDzSNMbMCuAMXP1lE_XiCQRy -O {large_root}/grokking_addition_full_run.pth")

def to_numpy(tensor, flat=False):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        # if isinstance(tensor[0])
        return np.array(tensor)
    elif isinstance(tensor, torch.Tensor):
        if flat:
            return tensor.flatten().detach().cpu().numpy()
        else:
            return tensor.detach().cpu().numpy()
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")

# Plotly bug means we need to write a graph to PDF first!
# https://github.com/plotly/plotly.py/issues/3469
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
# fig.show("vscode+colab")
fig.write_image("random.pdf")

import copy
pio.templates['grokking'] = copy.deepcopy(pio.templates['plotly'])
pio.templates.default = 'grokking'

pio.templates['grokking']['layout']['font']['family'] = 'Computer Modern'
pio.templates['grokking']['layout']['title'].update(dict(
    yref='paper',
    yanchor='bottom',
    y=1.,
    pad_b=10,
    xanchor='center',
    x=0.5,
    font_size=20,
))

pio.templates['grokking']['layout']['legend'].update(
    font_size = 12,
)
axis_dict = dict(
    title_font_size=15,
    tickfont_size=12,
    title_standoff=1.,
)
coloraxis_dict = dict(colorbar_x=1.01, 
                      colorbar_xanchor="left", 
                      colorbar_xpad=0)
pio.templates['grokking']['layout']['xaxis'].update(axis_dict)
pio.templates['grokking']['layout']['yaxis'].update(axis_dict)
pio.templates['grokking']['layout']['coloraxis'].update(coloraxis_dict)

# Adapt my general plotting code to specific grokking useful functions
def imshow_base_flatten(array, **kwargs):
    array = to_numpy(array)
    if array.shape[0]==p*p:
        array = einops.rearrange(array, "(p1 p2) ... -> p1 p2 ...", p1=p, p2=p)
    return imshow_base(array, **kwargs)


imshow = partial(imshow_base_flatten, color_continuous_scale='RdBu', color_continuous_midpoint=0.0, aspect='auto')
imshow_pos = partial(imshow_base_flatten, color_continuous_scale='Blues', aspect='auto')
inputs_heatmap = partial(imshow, xaxis='Input 1', yaxis='Input 2',
                         color_continuous_scale='RdBu', color_continuous_midpoint=0.0)
lines = line

def imshow_fourier(tensor, title='', animation_name='snapshot', facet_labels=[], return_fig=False, **kwargs):
    # Set nice defaults for plotting functions in the 2D fourier basis
    # tensor is assumed to already be in the Fourier Basis
    if tensor.shape[0]==p*p:
        tensor = unflatten_first(tensor)
    tensor = torch.squeeze(tensor)
    fig=px.imshow(to_numpy(tensor),
            x=fourier_basis_names, 
            y=fourier_basis_names, 
            labels={'x':'x Component', 
                    'y':'y Component', 
                    'animation_frame':animation_name},
            title=title,
            color_continuous_midpoint=0., 
            color_continuous_scale='RdBu', 
            **kwargs)
    fig.update(data=[{'hovertemplate':"%{x}x * %{y}y<br>Value:%{z:.4f}"}])
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    fig = fig
    if return_fig:
        return fig
    else:
        fig.show("vscode+colab")

def embed_to_cos_sin(fourier_embed):
    if len(fourier_embed.shape) == 1:
        return torch.stack([fourier_embed[1::2], fourier_embed[2::2]])
    else:
        return torch.stack([fourier_embed[:, 1::2], fourier_embed[:, 2::2]], dim=1)


def plot_embed_bars(fourier_embed, title='Norm of embedding of each Fourier Component', return_fig=False, **kwargs):
    cos_sin_embed = embed_to_cos_sin(fourier_embed)
    df = melt(cos_sin_embed)
    # display(df)
    group_labels = {0: 'sin', 1: 'cos'}
    df['Trig'] = df['0'].map(lambda x: group_labels[x])
    fig = px.bar(df, barmode='group', color='Trig', x='1', y='value', labels={
                 '1': '$w_k$', 'value': 'Norm'}, title=title, **kwargs)
    fig.update_layout(dict(legend_title=""))

    if return_fig:
        return fig
    else:
        fig.show("vscode+colab")



# write_image(fig, 'norm_fourier_embedding')
image_dir = high_level_root/'images'
json_dir = high_level_root/'jsons'
html_dir = high_level_root/'htmls'
big_latex_string = []
all_figure_names = []
def write_image(fig, name, file_type='pdf', apply_template=True, caption='', interpretation=''):
    pass
    # fig.show("vscode+colab")
    # html = fig.to_html(include_plotlyjs='cdn')
    # fig.write_html(html_dir/f"{name}.html")
    # print(html)


def cross_entropy_high_precision(logits, labels):
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly 
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes 
    # and dodgy gradients
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss

def full_loss(model, data):
    # Take the final position only
    logits = model(data)[:, -1]
    labels = torch.tensor([fn(i, j) for i, j, _ in data]).to('cuda')
    return cross_entropy_high_precision(logits, labels)

def test_logits(logits, bias_correction=False, original_logits=None, mode='all'):
    # Calculates cross entropy loss of logits representing a batch of all p^2 
    # possible inputs
    # Batch dimension is assumed to be first
    if logits.shape[1]==p*p:
        logits = logits.T
    if logits.shape==torch.Size([p*p, p+1]):
        logits = logits[:, :-1]
    logits = logits.reshape(p*p, p)
    if bias_correction:
        # Applies bias correction - we correct for any missing bias terms, 
        # independent of the input, by centering the new logits along the batch 
        # dimension, and then adding the average original logits across all inputs
        logits = einops.reduce(original_logits - logits, 'batch ... -> ...', 'mean') + logits
    if mode=='train':
        return cross_entropy_high_precision(logits[is_train], labels[is_train])
    elif mode=='test':
        return cross_entropy_high_precision(logits[is_test], labels[is_test])
    elif mode=='all':
        return cross_entropy_high_precision(logits, labels)

def unflatten_first(tensor):
    if tensor.shape[0]==p*p:
        return einops.rearrange(tensor, '(x y) ... -> x y ...', x=p, y=p)
    else: 
        return tensor
def cos(x, y):
    return (x.dot(y))/x.norm()/y.norm()
def mod_div(a, b):
    return (a*pow(b, p-2, p))%p
def normalize(tensor, axis=0):
    return tensor/(tensor).pow(2).sum(keepdim=True, axis=axis).sqrt()
def extract_freq_2d(tensor, freq):
    # Takes in a pxpx... or batch x ... tensor, returns a 3x3x... tensor of the 
    # Linear and quadratic terms of frequency freq
    tensor = unflatten_first(tensor)
    # Extracts the linear and quadratic terms corresponding to frequency freq
    index_1d = [0, 2*freq-1, 2*freq]
    # Some dumb manipulation to use fancy array indexing rules
    # Gets the rows and columns in index_1d
    return tensor[[[i]*3 for i in index_1d], [index_1d]*3]
def get_cov(tensor, norm=True):
    # Calculate covariance matrix
    if norm:
        tensor = normalize(tensor, axis=1)
    return tensor @ tensor.T
def is_close(a, b):
    return ((a-b).pow(2).sum()/(a.pow(2).sum().sqrt())/(b.pow(2).sum().sqrt())).item()

p=113
fourier_basis = []
fourier_basis.append(torch.ones(p)/np.sqrt(p))
fourier_basis_names = ['Const']
# Note that if p is even, we need to explicitly add a term for cos(kpi), ie 
# alternating +1 and -1
for i in range(1, p//2 +1):
    fourier_basis.append(torch.cos(2*torch.pi*torch.arange(p)*i/p))
    fourier_basis.append(torch.sin(2*torch.pi*torch.arange(p)*i/p))
    fourier_basis[-2]/=fourier_basis[-2].norm()
    fourier_basis[-1]/=fourier_basis[-1].norm()
    fourier_basis_names.append(f'cos {i}')
    fourier_basis_names.append(f'sin {i}')
fourier_basis = torch.stack(fourier_basis, dim=0).to('cuda')
# animate_lines(fourier_basis, snapshot_index=fourier_basis_names, snapshot='Fourier Component', title='Graphs of Fourier Components (Use Slider)')

def fft1d(tensor):
    # Converts a tensor with dimension p into the Fourier basis
    return tensor @ fourier_basis.T

def fourier_2d_basis_term(x_index, y_index):
    # Returns the 2D Fourier basis term corresponding to the outer product of 
    # the x_index th component in the x direction and y_index th component in the 
    # y direction
    # Returns a 1D vector of length p^2
    return (fourier_basis[x_index][:, None] * fourier_basis[y_index][None, :]).flatten()

def fft2d(mat):
    # Converts a pxpx... or batch x ... tensor into the 2D Fourier basis.
    # Output has the same shape as the original
    shape = mat.shape
    mat = einops.rearrange(mat, '(x y) ... -> x y (...)', x=p, y=p)
    fourier_mat = torch.einsum('xyz,fx,Fy->fFz', mat, fourier_basis, fourier_basis)
    return fourier_mat.reshape(shape)

def analyse_fourier_2d(tensor, top_k=10):
    # Processes a (p,p) or (p*p) tensor in the 2D Fourier Basis, showing the 
    # top_k terms and how large a fraction of the variance they explain
    values, indices = tensor.flatten().pow(2).sort(descending=True)
    rows = []
    total = values.sum().item()
    for i in range(top_k):
        rows.append([tensor.flatten()[indices[i]].item(),
                     values[i].item()/total, 
                     values[:i+1].sum().item()/total, 
                     fourier_basis_names[indices[i].item()//p], 
                     fourier_basis_names[indices[i]%p]])
    display(pd.DataFrame(rows, columns=['Coefficient', 'Frac explained', 'Cumulative frac explained', 'x', 'y']))

def get_2d_fourier_component(tensor, x, y):
    # Takes in a batch x ... tensor and projects it onto the 2D Fourier Component 
    # (x, y)
    vec = fourier_2d_basis_term(x, y).flatten()
    return vec[:, None] @ (vec[None, :] @ tensor)

def get_component_cos_xpy(tensor, freq, collapse_dim=False):
    # Gets the component corresponding to cos(freq*(x+y)) in the 2D Fourier basis
    # This is equivalent to the matrix cos((x+y)*freq*2pi/p)
    cosx_cosy_direction = fourier_2d_basis_term(2*freq-1, 2*freq-1).flatten()
    sinx_siny_direction = fourier_2d_basis_term(2*freq, 2*freq).flatten()
    # Divide by sqrt(2) to ensure it remains normalised
    cos_xpy_direction = (cosx_cosy_direction - sinx_siny_direction)/np.sqrt(2)
    # Collapse_dim says whether to project back into R^(p*p) space or not
    if collapse_dim:
        return (cos_xpy_direction @ tensor)
    else:
        return cos_xpy_direction[:, None] @ (cos_xpy_direction[None, :] @ tensor)

def get_component_sin_xpy(tensor, freq, collapse_dim=False):
    # Gets the component corresponding to sin((x+y)*freq*2pi/p) in the 2D Fourier basis
    sinx_cosy_direction = fourier_2d_basis_term(2*freq, 2*freq-1).flatten()
    cosx_siny_direction = fourier_2d_basis_term(2*freq-1, 2*freq).flatten()
    sin_xpy_direction = (sinx_cosy_direction + cosx_siny_direction)/np.sqrt(2)
    if collapse_dim:
        return (sin_xpy_direction @ tensor)
    else:
        return sin_xpy_direction[:, None] @ (sin_xpy_direction[None, :] @ tensor)

lr=1e-3 #@param
weight_decay = 1.0 #@param
p=113 #@param
d_model = 128 #@param
fn_name = 'add' #@param ['add', 'subtract', 'x2xyy2','rand']
frac_train = 0.3 #@param
num_epochs = 50000 #@param
save_models = False #@param
save_every = 100 #@param
# Stop training when test loss is <stopping_thresh
stopping_thresh = -1 #@param
seed = 0 #@param

num_layers = 1
batch_style = 'full'
d_vocab = p+1
n_ctx = 3
d_mlp = 4*d_model
num_heads = 4
assert d_model % num_heads == 0
d_head = d_model//num_heads
act_type = 'ReLU' #@param ['ReLU', 'GeLU']
# batch_size = 512
use_ln = False
random_answers = np.random.randint(low=0, high=p, size=(p, p))
fns_dict = {'add': lambda x,y:(x+y)%p, 'subtract': lambda x,y:(x-y)%p, 'x2xyy2':lambda x,y:(x**2+x*y+y**2)%p, 'rand':lambda x,y:random_answers[x][y]}
fn = fns_dict[fn_name]

def gen_train_test(frac_train, num, seed=0):
    # Generate train and test split
    pairs = [(i, j, num) for i in range(num) for j in range(num)]
    random.seed(seed)
    random.shuffle(pairs)
    div = int(frac_train*len(pairs))
    return pairs[:div], pairs[div:]

train, test = gen_train_test(frac_train, p, seed)

# Creates an array of Boolean indices according to whether each data point is in 
# train or test
# Used to index into the big batch of all possible data
is_train = []
is_test = []
for x in range(p):
    for y in range(p):
        if (x, y, 113) in train:
            is_train.append(True)
            is_test.append(False)
        else:
            is_train.append(False)
            is_test.append(True)
is_train = np.array(is_train)
is_test = np.array(is_test)   

full_run_data = torch.load(large_file_root/'grokking_addition_full_run.pth')

train_losses = full_run_data['train_losses'][:40000]
test_losses = full_run_data['test_losses'][:4000]

class Transformer(nn.Module):
    def __init__(self, num_layers, d_vocab, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, use_cache=False, use_ln=True):
        super().__init__()
        self.cache = {}
        self.use_cache = use_cache

        self.embed = Embed(d_vocab, d_model)
        self.pos_embed = PosEmbed(n_ctx, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model=[self]) for i in range(num_layers)])
        # self.ln = LayerNorm(d_model, model=[self])
        self.unembed = Unembed(d_vocab, d_model)
        self.use_ln = use_ln

        for name, module in self.named_modules():
            if type(module)==HookPoint:
                module.give_name(name)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        # x = self.ln(x)
        x = self.unembed(x)
        return x

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache
    
    def hook_points(self):
        return [module for name, module in self.named_modules() if 'hook' in name]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks('fwd')
            hp.remove_hooks('bwd')
    
    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()
        def save_hook_back(tensor, name):
            cache[name+'_grad'] = tensor[0].detach()
        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')

model = Transformer(num_layers=num_layers, d_vocab=d_vocab, d_model=d_model, d_mlp=d_mlp, d_head=d_head, num_heads=num_heads, n_ctx=n_ctx, act_type=act_type, use_cache=False, use_ln=use_ln)
model.to('cuda')
model.load_state_dict(full_run_data['state_dicts'][400])


# Helper variables
W_O = einops.rearrange(model.blocks[0].attn.W_O, 'm (i h)->i m h', i=num_heads)
W_K = model.blocks[0].attn.W_K
W_Q = model.blocks[0].attn.W_Q
W_V = model.blocks[0].attn.W_V
W_in = model.blocks[0].mlp.W_in
W_out = model.blocks[0].mlp.W_out
W_pos = model.pos_embed.W_pos.T
# We remove the equals sign dimension from the Embed and Unembed, so we can 
# apply a Fourier Transform over R^p
W_E = model.embed.W_E[:, :-1]
W_U = model.unembed.W_U[:, :-1].T

# The initial value of the residual stream at position 2 - constant for all inputs
final_pos_resid_initial = model.embed.W_E[:, -1] + W_pos[:, 2]

W_attn = torch.einsum('m,ihm,ihM,Mv->iv', final_pos_resid_initial, W_Q, W_K, W_E)
W_L = W_U @ W_out
W_neur = W_in @ W_O @ W_V @ W_E

all_data = torch.tensor([(i, j, p) for i in range(p) for j in range(p)]).to('cuda')
labels = torch.tensor([fn(i, j) for i, j, _ in all_data]).to('cuda')
cache = {}
model.remove_all_hooks()
model.cache_all(cache)
# Final position only
original_logits = model(all_data)[:, -1]
# Remove equals sign from output logits
original_logits = original_logits[:, :-1]
original_loss = cross_entropy_high_precision(original_logits, labels)
# Extracts out key activations
attn_mat = cache['blocks.0.attn.hook_attn'][:, :, 2, :2]
neuron_acts = cache['blocks.0.mlp.hook_post'][:, -1]
neuron_acts_pre = cache['blocks.0.mlp.hook_pre'][:, -1]

indices = []
for freq in range(1, p//2+1):
    index = []
    index.append(2*freq-1)
    index.append(2*freq)
    index.append(((2*freq - 1)*p)+0)
    index.append(((2*freq)*p)+0)
    index.append(((2*freq - 1)*p)+2*freq-1)
    index.append(((2*freq - 1)*p)+2*freq)
    index.append(((2*freq)*p)+2*freq-1)
    index.append(((2*freq)*p)+2*freq)
    indices.append(index)
indices = np.array(indices)


neuron_acts_centered = neuron_acts - einops.reduce(neuron_acts, 'batch neuron -> 1 neuron', 'mean')
# Note that fourier_neuron_acts[(0, 0), i]==0 for all i, because we centered the activations
fourier_neuron_acts = fft2d(neuron_acts_centered)
fourier_neuron_acts_square = fourier_neuron_acts.reshape(p, p, d_mlp)

neuron_norms = fourier_neuron_acts.pow(2).sum(0)
# print(neuron_norms.shape)

freq_acts = fourier_neuron_acts[indices.flatten()].reshape(56, 8, 512)
neuron_explanation = freq_acts[:].pow(2).sum(1)/neuron_norms
neuron_frac_explained = neuron_explanation.max(0).values
neuron_freqs = neuron_explanation.argmax(0)+1
neuron_freqs_original = neuron_freqs.clone()

key_freqs, neuron_freq_counts = np.unique(to_numpy(neuron_freqs), return_counts=True)

# To represent that they are in a special sixth cluster, we set the 
# frequency of these neurons to -1
neuron_freqs[neuron_frac_explained < 0.85] = -1.
neuron_freqs = to_numpy(neuron_freqs)
key_freqs_plus = np.concatenate([key_freqs, np.array([-1])])

neuron_labels_by_cluster = np.concatenate([np.arange(d_mlp)[neuron_freqs==freq] for freq in key_freqs_plus])

key_indices = []
for freq in key_freqs:
    index = []
    index.append(2*freq-1)
    index.append(2*freq)
    index.append(((2*freq - 1)*p)+0)
    index.append(((2*freq)*p)+0)
    index.append(((2*freq - 1)*p)+2*freq-1)
    index.append(((2*freq - 1)*p)+2*freq)
    index.append(((2*freq)*p)+2*freq-1)
    index.append(((2*freq)*p)+2*freq)
    key_indices.append(index)
key_indices = np.array(key_indices)

x_vec = torch.arange(p)[:, None, None].float().to("cuda")
y_vec = torch.arange(p)[None, :, None].float().to("cuda")
z_vec = torch.arange(p)[None, None, :].float().to("cuda")

# Sum of the true answer, uniformly
coses = []
for w in range(1, p//2 + 1):
    coses.append(torch.cos(w * torch.pi*2 / p * (x_vec + y_vec - z_vec)).to("cuda"))
coses = torch.stack(coses, axis=0).reshape(p//2, p*p, p)
coses/=coses.pow(2).sum([-2, -1], keepdim=True).sqrt()
# for i in range(3):
#     imshow(new_cube[:, :, i])

epochs = full_run_data['epochs'][:400]
metric_cache = {}
plot_metric = partial(lines, x=epochs, xaxis='Epoch', log_y=True)


def get_metrics(model, metric_cache, metric_fn, name, reset=False):
    if reset or (name not in metric_cache) or (len(metric_cache[name])==0):
        metric_cache[name]=[]
        for c, sd in enumerate(tqdm.tqdm((full_run_data['state_dicts'][:400]))):
            model.remove_all_hooks()
            model.load_state_dict(sd)
            out = metric_fn(model)
            if type(out)==torch.Tensor:
                out = to_numpy(out)
            metric_cache[name].append(out)
        model.load_state_dict(full_run_data['state_dicts'][400])
        try:
            metric_cache[name] = torch.tensor(metric_cache[name])
        except:
            metric_cache[name] = torch.tensor(np.array(metric_cache[name]))
def test_loss(model):
    logits = model(all_data)[:, -1, :-1]
    return test_logits(logits, False, mode='test')
# get_metrics(model, metric_cache, test_loss, 'test_loss')
def train_loss(model):
    logits = model(all_data)[:, -1, :-1]
    return test_logits(logits, False, mode='train')
# get_metrics(model, metric_cache, train_loss, 'train_loss')

def acc(logits, mode='all'):
    bool_vec = (logits.argmax(1)==labels)
    if mode=='all':
        subset=None 
    elif mode=='train':
        subset = is_train 
    elif mode=='test':
        subset = is_test 
    else:
        raise ValueError("Invalid mode")
    if subset is not None:
        bool_vec = bool_vec[subset]
    return bool_vec.sum()/len(bool_vec)

def get_train_acc(model):
    logits = model(all_data)[:, -1, :-1]
    # def acc(logits):
    # return (logits.argmax(1)==labels).sum()/p/p
    bool_vec = logits.argmax(1)[is_train] == labels[is_train]
    return bool_vec.sum()/len(bool_vec)
# get_metrics(model, metric_cache, get_train_acc, 'train_acc')
# plot_metric([metric_cache['train_acc']], log_y=False)

def get_test_acc(model):
    logits = model(all_data)[:, -1, :-1]
    # def acc(logits):
    # return (logits.argmax(1)==labels).sum()/p/p
    bool_vec = logits.argmax(1)[is_test] == labels[is_test]
    return bool_vec.sum()/len(bool_vec)
# get_metrics(model, metric_cache, get_test_acc, 'test_acc')

# Construct a mask that has a 1 on the quadratic terms of a specific frequency, 
# and zeros everywhere else
quadratic_mask = torch.zeros((p, p), device='cuda')
for freq in range(1, (p//2)+1):
    for i in [2*freq-1, 2*freq]:
        for j in [2*freq-1, 2*freq]:
            quadratic_mask[i, j]=1.
quadratic_mask = einops.rearrange(quadratic_mask, 'x y->(x y) 1')
# imshow_fourier(quadratic_mask)
square_quadratic_mask = einops.rearrange(quadratic_mask, '(x y) 1->x y 1', x=p, y=p)

key_freq_strs = list(map(str, key_freqs))
def calculate_excluded_loss_2D(model):
    logits = model(all_data)[:, -1, :-1]
    row = []
    for freq in range(1, p//2+1):
        row.append(test_logits((logits - 
                                get_component_cos_xpy(logits, freq) - 
                                get_component_sin_xpy(logits, freq)), 
                               bias_correction=False, 
                               mode='train').item())
    return row
# get_metrics(model, metric_cache, calculate_excluded_loss_2D, 'excluded_loss_2D', reset=False)

def calculate_excluded_loss_2D_full(model):
    logits = model(all_data)[:, -1, :-1]
    new_logits = logits.clone()
    row = []
    for freq in key_freqs:
        new_logits -= (get_component_cos_xpy(logits, freq))
        new_logits -= (get_component_sin_xpy(logits, freq))
        
    return test_logits(new_logits, False, mode='train')
# get_metrics(model, metric_cache, calculate_excluded_loss_2D_full, 'excluded_loss_2D_full', reset=False)

def calculate_excluded_acc_2D(model):
    logits = model(all_data)[:, -1, :-1]
    row = []
    for freq in range(1, p//2+1):
        row.append(acc((logits - 
                                get_component_cos_xpy(logits, freq) - 
                                get_component_sin_xpy(logits, freq)), 
                               mode='train').item())
    return row
# get_metrics(model, metric_cache, calculate_excluded_acc_2D, 'excluded_acc_2D', reset=False)

def calculate_excluded_acc_2D_full(model):
    logits = model(all_data)[:, -1, :-1]
    new_logits = logits.clone()
    row = []
    for freq in key_freqs:
        new_logits -= (get_component_cos_xpy(logits, freq))
        new_logits -= (get_component_sin_xpy(logits, freq))
        
    return acc(new_logits, mode='train')
# get_metrics(model, metric_cache, calculate_excluded_acc_2D_full, 'excluded_acc_2D_full', reset=False)


def excluded_loss_3D(model):
    logits = model(all_data)[:, -1, :-1]
    vals = ((coses * logits[None, :]).sum([-2, -1]))

    return [test_logits(logits - (v * coses[c]), mode='train').item() for c, v in enumerate(vals)]
# get_metrics(model, metric_cache, excluded_loss_3D, 'excluded_loss_3D', reset=False)


def excluded_loss_3D_full(model):
    logits = model(all_data)[:, -1, :-1]
    vals = ((coses * logits[None, :]).sum([-2, -1]))
    logits = logits - (vals[key_freqs-1, None, None] * coses[key_freqs-1]).sum(0)
    return test_logits(logits, mode='train')
# get_metrics(model, metric_cache, excluded_loss_3D_full, 'excluded_loss_3D_full', reset=False)


def excluded_acc_3D(model):
    logits = model(all_data)[:, -1, :-1]
    vals = ((coses * logits[None, :]).sum([-2, -1]))

    return [acc(logits - (v * coses[c]), mode='train').item() for c, v in enumerate(vals)]
# get_metrics(model, metric_cache, excluded_acc_3D, 'excluded_acc_3D', reset=False)


def excluded_acc_3D_full(model):
    logits = model(all_data)[:, -1, :-1]
    vals = ((coses * logits[None, :]).sum([-2, -1]))
    logits = logits - (vals[key_freqs-1, None, None] * coses[key_freqs-1]).sum(0)
    return acc(logits, mode='train')
# get_metrics(model, metric_cache, excluded_acc_3D_full, 'excluded_acc_3D_full', reset=False)

def trig_loss(model, mode='all'):
    logits = model(all_data)[:, -1, :-1]
    trig_logits = sum([get_component_cos_xpy(logits, freq) + 
                   get_component_sin_xpy(logits, freq)
                   for freq in key_freqs])
    return test_logits(trig_logits, 
                       bias_correction=True, 
                       original_logits=logits, 
                       mode=mode)
# get_metrics(model, metric_cache, trig_loss, 'trig_loss')

trig_loss_train = partial(trig_loss, mode='train')
# get_metrics(model, metric_cache, trig_loss_train, 'trig_loss_train')

def trig_acc(model, mode='all'):
    logits = model(all_data)[:, -1, :-1]
    trig_logits = sum([get_component_cos_xpy(logits, freq) + 
                   get_component_sin_xpy(logits, freq)
                   for freq in key_freqs])
    trig_logits += original_logits.mean(0, keepdim=True) - trig_logits.mean(0, keepdim=True)
    return acc(trig_logits, mode=mode)
# get_metrics(model, metric_cache, trig_acc, 'trig_acc', reset=False)

trig_acc_train = partial(trig_acc, mode='train')
# get_metrics(model, metric_cache, trig_acc_train, 'trig_acc_train', reset=False)

parameter_names = [name for name, param in model.named_parameters()]
def sum_sq_weights(model):
    row = []
    for name, param in model.named_parameters():
        row.append(param.pow(2).sum().item())
    return row
# get_metrics(model, metric_cache, sum_sq_weights, 'sum_sq_weights')


def get_cos_coeffs(model):
    logits = model(all_data)[:, -1, :-1]
    vals = ((coses * logits[None, :]).sum([-2, -1]))

    return vals
# get_metrics(model, metric_cache, get_cos_coeffs, 'cos_coeffs')

def get_cos_sim(model):
    logits = model(all_data)[:, -1, :-1]
    vals = ((coses * logits[None, :]).sum([-2, -1]))

    return vals/logits.norm()
# get_metrics(model, metric_cache, get_cos_sim, 'cos_sim')

def get_fourier_embedding(model):
    return (model.embed.W_E[:, :-1] @ fourier_basis.T).norm(dim=0)
# get_metrics(model, metric_cache, get_fourier_embedding, 'fourier_embedding')
