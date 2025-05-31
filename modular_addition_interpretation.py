from modular_addition_utils import *
from toolz.curried import *
import interpretation
from typing import Optional, Literal
from sklearn.linear_model import LinearRegression
from helpers import compose_reversed
from math import cos, sin, pi

wk = lambda i: (2 * i * pi)/p
wkx = torch.cat([(wk(f) * torch.arange(p)).unsqueeze(1) for f in key_freqs], dim=1)
cos_wkx = torch.cos(wkx)
sin_wkx = torch.sin(wkx)

wkxs = [wk(f) * torch.arange(p) for f in key_freqs]
fourier_components = torch.cat([torch.cat([torch.cos(wkx).unsqueeze(1), torch.sin(wkx).unsqueeze(1)], dim=1) for wkx in wkxs], dim=1)

device = torch.device("cuda:0")

tok_map = {
    i: i
    for i in range(113)
} | {'=': 113}
rev_tok_map = {v: k for k,v in tok_map.items()}

wk = lambda i: (2 * i * pi)/p
wkx = torch.cat([(wk(f) * torch.arange(p)).unsqueeze(1) for f in key_freqs], dim=1)
cos_wkx = torch.cos(wkx)
sin_wkx = torch.sin(wkx)

wkxs = [wk(f) * torch.arange(p) for f in key_freqs]
fourier_components = torch.cat([torch.cat([torch.cos(wkx).unsqueeze(1), torch.sin(wkx).unsqueeze(1)], dim=1) for wkx in wkxs], dim=1)

X = W_E.T.detach().cpu().numpy()
Y = fourier_components.cpu().numpy()

alpha_1_model = LinearRegression().fit(X, Y)
component_w = torch.Tensor(alpha_1_model.coef_)
component_b = torch.Tensor(alpha_1_model.intercept_)

gamma_1_model = LinearRegression().fit(Y, X)
rep_w = torch.Tensor(gamma_1_model.coef_).to(device)
rep_b = torch.Tensor(gamma_1_model.intercept_).to(device)

identity = lambda x, *args, **kwargs: x

def round(tensor: torch.Tensor, decimals: Optional[int] = 2, nth: Optional[int] = None) -> torch.Tensor:
    """Round to decimals decimals if set, nth overrides and instead rounds to nearest 1/nth."""
    if nth is None:
        if decimals is None:
            return tensor
    
        return tensor.round(decimals=decimals)
    else:
        return (nth * tensor).round(decimals=0) / nth

def gamma_1(fourier_components: torch.Tensor) -> torch.Tensor:
    input_embeds = rep_b + torch.einsum("bic, ec -> bie", fourier_components.to(device), rep_w)
    embeds = torch.cat([input_embeds, model.embed.W_E[:,tok_map['=']].unsqueeze(0).tile(input_embeds.shape[0],1,1)], dim=1)
    return embeds

def alpha_1(embeds: torch.Tensor, decimals: Optional[int] = 2, nth: Optional[int] = None) -> torch.Tensor:
    components = component_b + torch.einsum("bie, ce -> bic", embeds[:,:-1,:], component_w)
    return round(components, decimals, nth)

## Concrete model components
def tok_embed(toks: list) -> torch.Tensor:
    return model.embed(toks)

def pos_embed_mlp_hidden(embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    attn_in = model.pos_embed(embeds)
    resid = attn_in + model.blocks[0].attn(attn_in)

    mlp_mid = torch.einsum("md, bpd -> bpm", W_in, resid) + model.blocks[0].mlp.b_in
    if act_type == "ReLU":
        mlp_mid = F.relu(mlp_mid)
    elif act_type == "GeLU":
        mlp_mid = F.gelu(mlp_mid)

    return (mlp_mid, resid)

def mlp_out_unembed(mlp_hidden_residual: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    mlp_mid, resid  = mlp_hidden_residual
    mlp_out = resid + torch.einsum("dm, bpm -> bpd", W_out, mlp_mid) + model.blocks[0].mlp.b_out
    logits = model.unembed(mlp_out)
    return logits[:,-1].argmax(dim=-1)

## Abstract model: first two components
def calculate_fourier_components(toks: list, decimals: Optional[int] = 2, nth: Optional[int] = None) -> torch.Tensor:
    """
    Returns the components of a in outputs[:,0] and of b in outputs[:,1].
    The components are ordered e.g. cos(w_0 a), sin(w_0 a), cos(w_1 a), ...
    """
    return round(
        torch.cat(
            [
                fourier_components[list(ts[:2])].unsqueeze(0)
                for ts in toks
            ], 
            dim=0,
        ), 
        decimals,
        nth,
    )

# Prior to mapping outputs to equivalence classes
def angle_sum_identities(fourier_components: torch.Tensor, **kwargs) -> torch.Tensor:
    """Returns cos(w_k(a+b)) in outputs[:,0], sin(w_k(a+b)) in outputs[:,1]"""
    a = fourier_components[:,0,:]
    b = fourier_components[:,1,:]

    cos_a = a[...,::2]
    sin_a = a[...,1::2]
    cos_b = b[...,::2]
    sin_b = b[...,1::2]

    cos_ab = cos_a * cos_b - sin_a * sin_b
    sin_ab = sin_a * cos_b + cos_a * sin_b

    return torch.cat([cos_ab.unsqueeze(1), sin_ab.unsqueeze(1)], dim=1)
    
def to_tensor(classes: list) -> torch.Tensor:
    return torch.cat([torch.Tensor(c if isinstance(c, torch.Tensor) else c.item).unsqueeze(0) for c in classes], dim=0)
    
def argmax_difference_identity(angle_sums: list, **kwargs) -> list[int]:
    angle_sums = to_tensor(angle_sums)
    
    cos_wkab = angle_sums[:,0].to(device)
    sin_wkab = angle_sums[:,1].to(device)
    
    diff_identity = torch.einsum("tf, bf -> btf", cos_wkx.to(device), cos_wkab) +  torch.einsum("tf, bf -> btf", sin_wkx.to(device), sin_wkab)
    sum_freqs = diff_identity.sum(dim=-1)
    output = sum_freqs.argmax(dim=-1)

    return output

## Interpretation; the alphas and gammas will be assigned below
abstract_model_layers = [calculate_fourier_components, angle_sum_identities]
concrete_model_layers = [tok_embed, pos_embed_mlp_hidden, mlp_out_unembed]
alphas = [identity, alpha_1, identity, identity]
gammas = [identity, gamma_1, identity, identity]

## Generate interpretation and evaluation utilities
def interp_tools_generator(
    abstract_model_layers=abstract_model_layers,
    concrete_model_layers=concrete_model_layers,
    alphas=alphas,
    gammas=gammas,
):
    def generation_fn(
        decimals: Optional[int] = 3, 
        nth: Optional[int] = None,
    ):
        def intermediates(input, model=model):
            cache = {}
            model.remove_all_hooks()
            model.cache_all(cache)
            model_output = model(input)
        
            return cache
    
        abs_layers = [partial(f, decimals=decimals, nth=nth) for f in abstract_model_layers]
            
        interp_tools = {
            "abstract_model_layers": abs_layers,
            "concrete_model_layers": concrete_model_layers,
            "concrete_model": compose_reversed(concrete_model_layers),
            "abstract_model": lambda *args, decimals=decimals, nth=nth, **kwargs: compose_reversed([partial(l, decimals=decimals, nth=nth) for l in abstract_model_layers])(*args, **kwargs),
            "intermediates": intermediates,
            "alphas": [partial(f, decimals=decimals, nth=nth) for f in alphas],
            "model": model,
            "gammas": gammas,
            "train": train,
            "test": test,
        }
        interp_tools |= interpretation.get_axiom_models(**interp_tools)
        interp_tools |= interpretation.get_axiom_evaluators(**interp_tools)
        interp_tools["get_all_intermediates"] = lambda data, **kwargs: interpretation.get_all_intermediates(np.array(data), **(interp_tools | kwargs))

        return interp_tools

    return generation_fn

generate_interp_tools = interp_tools_generator()
interp_tools = generate_interp_tools()

ints = interp_tools["get_all_intermediates"](train)
mid_activations = ints["blocks.0.mlp.hook_post"][:,-1,:]
mean_activations = ints["blocks.0.mlp.hook_post"].mean(dim=0)
resid_mean = ints["blocks.0.hook_resid_mid"].mean(dim=0)

## Learn alpha_2 and gamma_2
X_2 = mid_activations.cpu().numpy()
Y_2 = interp_tools["abstract_model"](train, decimals=None).reshape(len(train), -1).cpu().numpy()

alpha_2_model = LinearRegression().fit(X_2, Y_2)
w_alpha_2 = torch.Tensor(alpha_2_model.coef_)
b_alpha_2 = torch.Tensor(alpha_2_model.intercept_)

gamma_2_model = LinearRegression().fit(Y_2, X_2)
w_gamma_2 = torch.Tensor(gamma_2_model.coef_).to(device)
b_gamma_2 = torch.Tensor(gamma_2_model.intercept_).to(device)

    
def gamma_2(sum_components: list) -> tuple[torch.Tensor, torch.Tensor]:
    sum_components = to_tensor(sum_components)
    numel = sum_components.shape[0]
    mids = mean_activations.to(device).unsqueeze(0).tile((numel, 1, 1))
    resids = resid_mean.to(device).unsqueeze(0).tile((numel, 1, 1))
    mlp_mid_output = b_gamma_2 + torch.einsum("bc, ec -> be", sum_components.to(device).reshape(numel, -1), w_gamma_2)
    mids[:,-1,:] = mlp_mid_output

    return mids, resids

## Define the second abstract component up to equivalence
class EqualityEquivalenceClass:
    def __init__(self, item: torch.Tensor):
        self.item = item

    def __eq__(self, other) -> bool:
        return torch.all(self.item == other.item)

class RoundEquivalenceClass:
    def __init__(self, item: torch.Tensor, decimals: int=1):
        self.item = round(item, decimals=decimals)

class ConcreteAndAbstractEquivalenceClass:
    def __init__(self, item: torch.Tensor):
        self.item = item

    def __eq__(self, other) -> bool:
        concrete_equivalent = mlp_out_unembed(gamma_2([self]))[0] == mlp_out_unembed(gamma_2([other]))[0]
        
        abstract_out = argmax_difference_identity([self])[0]
        abstract_out_other = argmax_difference_identity([other])[0]
        abstract_equivalent = abstract_out == abstract_out_other

        return concrete_equivalent and abstract_equivalent

def generate_interpretation_tools(*args, equivalence_class=ConcreteAndAbstractEquivalenceClass, **kwargs):
    def alpha_2(mids_resids: tuple[torch.Tensor, torch.Tensor], **kwargs) -> list[equivalence_class]:
        mids, _ = mids_resids
        output_mids = mids[:,-1,:]
        output_mids = b_alpha_2 + torch.einsum("sd, bd -> bs", w_alpha_2, output_mids)
        sum_components = output_mids.reshape(mids.shape[0], 2, -1)
        return [equivalence_class(sc) for sc in sum_components]
    
    def angle_sum_identities_equivalence(fourier_components: torch.Tensor, **kwargs) -> list[equivalence_class]:
        return [equivalence_class(asi) for asi in angle_sum_identities(fourier_components, **kwargs).cpu()]
        
    return interp_tools_generator(
        abstract_model_layers=[calculate_fourier_components, angle_sum_identities_equivalence, argmax_difference_identity],
        concrete_model_layers=[tok_embed, pos_embed_mlp_hidden, mlp_out_unembed],
        alphas=[identity, alpha_1, alpha_2, identity],
        gammas=[identity, gamma_1, gamma_2, identity],
    )(*args, **kwargs)