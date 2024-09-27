import os
import numpy as np # type: ignore
import torch # type: ignore
from transformers import GPT2LMHeadModel, GPT2Config # type: ignore


#You can change default file path 
filepath = "./src/"
filename = "gpt2_params.c"
fullpath = filepath + filename


def delete_file(filepath):
    if os.path.isfile(filepath):
        os.remove(filepath)

def save_weight_to_header(file_path, weights):
    with open(file_path, 'w') as f:
        f.write('#include "gpt2_params.h"\n\n')

        # Define arrays for weights
        f.write('// Define arrays for weights\n')

        # Define ln_1_w for all layers
        ln_1_w_layers = np.array(weights['ln_1_w'])
        ln_1_w_shape = ln_1_w_layers.shape
        f.write(f'const elem_t ln_1_w[{ln_1_w_shape[0]}][{ln_1_w_shape[1]}] = {{\n')
        for layer_index, layer in enumerate(ln_1_w_layers):
            row_str = ', '.join(map(str, layer.flatten()))
            f.write(f'    {{ {row_str} }}' + (',' if layer_index < len(ln_1_w_layers) - 1 else '') + '\n')
        f.write('};\n\n')

        # Define ln_1_b for all layers
        ln_1_b_layers = np.array(weights['ln_1_b'])
        ln_1_b_shape = ln_1_b_layers.shape
        f.write(f'const elem_t ln_1_b[{ln_1_b_shape[0]}][{ln_1_b_shape[1]}] = {{\n')
        for layer_index, layer in enumerate(ln_1_b_layers):
            row_str = ', '.join(map(str, layer.flatten()))
            f.write(f'    {{ {row_str} }}' + (',' if layer_index < len(ln_1_b_layers) - 1 else '') + '\n')
        f.write('};\n\n')

        # Define ln_2_w for all layers
        ln_2_w_layers = np.array(weights['ln_2_w'])
        ln_2_w_shape = ln_2_w_layers.shape
        f.write(f'const elem_t ln_2_w[{ln_2_w_shape[0]}][{ln_2_w_shape[1]}] = {{\n')
        for layer_index, layer in enumerate(ln_2_w_layers):
            row_str = ', '.join(map(str, layer.flatten()))
            f.write(f'    {{ {row_str} }}' + (',' if layer_index < len(ln_2_w_layers) - 1 else '') + '\n')
        f.write('};\n\n')

        # Define ln_2_b for all layers
        ln_2_b_layers = np.array(weights['ln_2_b'])
        ln_2_b_shape = ln_2_b_layers.shape
        f.write(f'const elem_t ln_2_b[{ln_2_b_shape[0]}][{ln_2_b_shape[1]}] = {{\n')
        for layer_index, layer in enumerate(ln_2_b_layers):
            row_str = ', '.join(map(str, layer.flatten()))
            f.write(f'    {{ {row_str} }}' + (',' if layer_index < len(ln_2_b_layers) - 1 else '') + '\n')
        f.write('};\n\n')

        # Define attn_query_w for all layers
        attn_query_w_layers = np.array(weights['attn_query_w'])
        attn_query_w_shape = attn_query_w_layers.shape
        f.write(f'const elem_t attn_query_w[{attn_query_w_shape[0]}][{attn_query_w_shape[1]}][{attn_query_w_shape[2]}] = {{\n')
        for layer_index, layer in enumerate(attn_query_w_layers):
            f.write('    {\n')
            for row_index, row in enumerate(layer):
                row_str = ', '.join(map(str, row))
                f.write(f'        {{ {row_str} }}' + (',' if row_index < len(layer) - 1 else '') + '\n')
            f.write('    }' + (',' if layer_index < len(attn_query_w_layers) - 1 else '') + '\n')
        f.write('};\n\n')

        # Define attn_query_b for all layers
        attn_query_b_layers = np.array(weights['attn_query_b'])
        attn_query_b_shape = attn_query_b_layers.shape
        f.write(f'const elem_t attn_query_b[{attn_query_b_shape[0]}][{attn_query_b_shape[1]}] = {{\n')
        for layer_index, layer in enumerate(attn_query_b_layers):
            row_str = ', '.join(map(str, layer.flatten()))
            f.write(f'    {{ {row_str} }}' + (',' if layer_index < len(attn_query_b_layers) - 1 else '') + '\n')
        f.write('};\n\n')

        # Define attn_key_w for all layers
        attn_key_w_layers = np.array(weights['attn_key_w'])
        attn_key_w_shape = attn_key_w_layers.shape
        f.write(f'const elem_t attn_key_w[{attn_key_w_shape[0]}][{attn_key_w_shape[1]}][{attn_key_w_shape[2]}] = {{\n')
        for layer_index, layer in enumerate(attn_key_w_layers):
            f.write('    {\n')
            for row_index, row in enumerate(layer):
                row_str = ', '.join(map(str, row))
                f.write(f'        {{ {row_str} }}' + (',' if row_index < len(layer) - 1 else '') + '\n')
            f.write('    }' + (',' if layer_index < len(attn_key_w_layers) - 1 else '') + '\n')
        f.write('};\n\n')

        # Define attn_key_b for all layers
        attn_key_b_layers = np.array(weights['attn_key_b'])
        attn_key_b_shape = attn_key_b_layers.shape
        f.write(f'const elem_t attn_key_b[{attn_key_b_shape[0]}][{attn_key_b_shape[1]}] = {{\n')
        for layer_index, layer in enumerate(attn_key_b_layers):
            row_str = ', '.join(map(str, layer.flatten()))
            f.write(f'    {{ {row_str} }}' + (',' if layer_index < len(attn_key_b_layers) - 1 else '') + '\n')
        f.write('};\n\n')

        # Define attn_value_w for all layers
        attn_value_w_layers = np.array(weights['attn_value_w'])
        attn_value_w_shape = attn_value_w_layers.shape
        f.write(f'const elem_t attn_value_w[{attn_value_w_shape[0]}][{attn_value_w_shape[1]}][{attn_value_w_shape[2]}] = {{\n')
        for layer_index, layer in enumerate(attn_value_w_layers):
            f.write('    {\n')
            for row_index, row in enumerate(layer):
                row_str = ', '.join(map(str, row))
                f.write(f'        {{ {row_str} }}' + (',' if row_index < len(layer) - 1 else '') + '\n')
            f.write('    }' + (',' if layer_index < len(attn_value_w_layers) - 1 else '') + '\n')
        f.write('};\n\n')

        # Define attn_value_b for all layers
        attn_value_b_layers = np.array(weights['attn_value_b'])
        attn_value_b_shape = attn_value_b_layers.shape
        f.write(f'const elem_t attn_value_b[{attn_value_b_shape[0]}][{attn_value_b_shape[1]}] = {{\n')
        for layer_index, layer in enumerate(attn_value_b_layers):
            row_str = ', '.join(map(str, layer.flatten()))
            f.write(f'    {{ {row_str} }}' + (',' if layer_index < len(attn_value_b_layers) - 1 else '') + '\n')
        f.write('};\n\n')

        # Define attn_proj_w for all layers
        attn_proj_w_layers = np.array(weights['attn_proj_w'])
        attn_proj_w_shape = attn_proj_w_layers.shape
        f.write(f'const elem_t attn_proj_w[{attn_proj_w_shape[0]}][{attn_proj_w_shape[1]}][{attn_proj_w_shape[2]}] = {{\n')
        for layer_index, layer in enumerate(attn_proj_w_layers):
            f.write('    {\n')
            for row_index, row in enumerate(layer):
                row_str = ', '.join(map(str, row))
                f.write(f'        {{ {row_str} }}' + (',' if row_index < len(layer) - 1 else '') + '\n')
            f.write('    }' + (',' if layer_index < len(attn_proj_w_layers) - 1 else '') + '\n')
        f.write('};\n\n')

        # Define attn_proj_b for all layers
        attn_proj_b_layers = np.array(weights['attn_proj_b'])
        attn_proj_b_shape = attn_proj_b_layers.shape
        f.write(f'const elem_t attn_proj_b[{attn_proj_b_shape[0]}][{attn_proj_b_shape[1]}] = {{\n')
        for layer_index, layer in enumerate(attn_proj_b_layers):
            row_str = ', '.join(map(str, layer.flatten()))
            f.write(f'    {{ {row_str} }}' + (',' if layer_index < len(attn_proj_b_layers) - 1 else '') + '\n')
        f.write('};\n\n')

        # Define mlp_fc_w for all layers
        mlp_fc_w_layers = np.array(weights['mlp_fc_w'])
        mlp_fc_w_shape = mlp_fc_w_layers.shape
        f.write(f'const elem_t mlp_fc_w[{mlp_fc_w_shape[0]}][{mlp_fc_w_shape[1]}][{mlp_fc_w_shape[2]}] = {{\n')
        for layer_index, layer in enumerate(mlp_fc_w_layers):
            f.write('    {\n')
            for row_index, row in enumerate(layer):
                row_str = ', '.join(map(str, row))
                f.write(f'        {{ {row_str} }}' + (',' if row_index < len(layer) - 1 else '') + '\n')
            f.write('    }' + (',' if layer_index < len(mlp_fc_w_layers) - 1 else '') + '\n')
        f.write('};\n\n')

        # Define mlp_fc_b for all layers
        mlp_fc_b_layers = np.array(weights['mlp_fc_b'])
        mlp_fc_b_shape = mlp_fc_b_layers.shape
        f.write(f'const elem_t mlp_fc_b[{mlp_fc_b_shape[0]}][{mlp_fc_b_shape[1]}] = {{\n')
        for layer_index, layer in enumerate(mlp_fc_b_layers):
            row_str = ', '.join(map(str, layer.flatten()))
            f.write(f'    {{ {row_str} }}' + (',' if layer_index < len(mlp_fc_b_layers) - 1 else '') + '\n')
        f.write('};\n\n')

        # Define mlp_proj_w for all layers
        mlp_proj_w_layers = np.array(weights['mlp_proj_w'])
        mlp_proj_w_shape = mlp_proj_w_layers.shape
        f.write(f'const elem_t mlp_proj_w[{mlp_proj_w_shape[0]}][{mlp_proj_w_shape[1]}][{mlp_proj_w_shape[2]}] = {{\n')
        for layer_index, layer in enumerate(mlp_proj_w_layers):
            f.write('    {\n')
            for row_index, row in enumerate(layer):
                row_str = ', '.join(map(str, row))
                f.write(f'        {{ {row_str} }}' + (',' if row_index < len(layer) - 1 else '') + '\n')
            f.write('    }' + (',' if layer_index < len(mlp_proj_w_layers) - 1 else '') + '\n')
        f.write('};\n\n')

        # Define mlp_proj_b for all layers
        mlp_proj_b_layers = np.array(weights['mlp_proj_b'])
        mlp_proj_b_shape = mlp_proj_b_layers.shape
        f.write(f'const elem_t mlp_proj_b[{mlp_proj_b_shape[0]}][{mlp_proj_b_shape[1]}] = {{\n')
        for layer_index, layer in enumerate(mlp_proj_b_layers):
            row_str = ', '.join(map(str, layer.flatten()))
            f.write(f'    {{ {row_str} }}' + (',' if layer_index < len(mlp_proj_b_layers) - 1 else '') + '\n')
        f.write('};\n\n')

        # Define ln_f_w 
        ln_f_w_shape = weights['ln_f_w'].shape
        f.write(f'const elem_t ln_f_w[{ln_f_w_shape[0]}] = {{\n')
        row_str = ', '.join(map(str, weights['ln_f_w'].flatten()))
        f.write(f'    {row_str}\n')
        f.write('};\n\n')

        # Define ln_f_b
        ln_f_b_shape = weights['ln_f_b'].shape
        f.write(f'const elem_t ln_f_b[{ln_f_b_shape[0]}] = {{\n')
        row_str = ', '.join(map(str, weights['ln_f_b'].flatten()))
        f.write(f'    {row_str}\n')
        f.write('};\n\n')
        
        # Define wte
        f.write('elem_t wte_t[768][50272];')
        wte_shape = weights['wte'].shape
        f.write(f'const elem_t wte[{wte_shape[0]}][{wte_shape[1]}] = {{\n')
        for layer_index, layer_weights in enumerate(weights['wte']):
            row_str = ', '.join(map(str, layer_weights.flatten()))
            f.write(f'    {{ {row_str} }}' + (',' if layer_index < len(weights['wte']) - 1 else '') + '\n')
        f.write('};\n\n')
        
        # Define wpe
        wpe_shape = weights['wpe'].shape
        f.write(f'const elem_t wpe[{wpe_shape[0]}][{wpe_shape[1]}] = {{\n')
        for layer_index, layer_weights in enumerate(weights['wpe']):
            row_str = ', '.join(map(str, layer_weights.flatten()))
            f.write(f'    {{ {row_str} }}' + (',' if layer_index < len(weights['wpe']) - 1 else '') + '\n')
        f.write('};\n\n')


def extract_weights(model):
    config = model.config
    weights = {}

    # Extract layer-wise parameters
    for i, layer in enumerate(model.transformer.h):
        hidden_size = config.hidden_size

        # Initialize weights for all layers
        weights.setdefault('ln_1_w', []).append(layer.ln_1.weight.data.numpy())
        weights.setdefault('ln_1_b', []).append(layer.ln_1.bias.data.numpy())
        attn_weights = layer.attn.c_attn.weight.data.numpy()
        
        weights.setdefault('attn_query_w', []).append(attn_weights[:hidden_size, :hidden_size])
        weights.setdefault('attn_query_b', []).append(layer.attn.c_attn.bias.data.numpy()[:hidden_size])
        weights.setdefault('attn_key_w', []).append(attn_weights[:hidden_size, hidden_size:2*hidden_size])
        weights.setdefault('attn_key_b', []).append(layer.attn.c_attn.bias.data.numpy()[hidden_size:2*hidden_size])
        weights.setdefault('attn_value_w', []).append(attn_weights[:hidden_size, 2*hidden_size:])
        weights.setdefault('attn_value_b', []).append(layer.attn.c_attn.bias.data.numpy()[2*hidden_size:])
        weights.setdefault('attn_proj_w', []).append(layer.attn.c_proj.weight.data.numpy())
        weights.setdefault('attn_proj_b', []).append(layer.attn.c_proj.bias.data.numpy())
        weights.setdefault('ln_2_w', []).append(layer.ln_2.weight.data.numpy())
        weights.setdefault('ln_2_b', []).append(layer.ln_2.bias.data.numpy())
        weights.setdefault('mlp_fc_w', []).append(layer.mlp.c_fc.weight.data.numpy())
        weights.setdefault('mlp_fc_b', []).append(layer.mlp.c_fc.bias.data.numpy())
        weights.setdefault('mlp_proj_w', []).append(layer.mlp.c_proj.weight.data.numpy())
        weights.setdefault('mlp_proj_b', []).append(layer.mlp.c_proj.bias.data.numpy())

    weights['ln_f_w'] = model.transformer.ln_f.weight.data.numpy()
    weights['ln_f_b'] = model.transformer.ln_f.bias.data.numpy()
    weights['wte'] = model.transformer.wte.weight.data.numpy()
    weights['wpe'] = model.transformer.wpe.weight.data.numpy()
    
    return weights

# Load pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Extract weights
weights = extract_weights(model)

# Save to header file
delete_file(fullpath)
save_weight_to_header(fullpath, weights)
