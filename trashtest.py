from src.utils.cfg_utils import init_dflts_to_of, get_dflts_to_of, set_if_not

local_config = {'parameters':{}}
# Gneral Configuration
set_if_not(local_config, 'lr', {'generator': 0.001,'discriminator': 0.01})                   

# Proto-model configuration:
#Declare the default classes to use
proto_kls='src.explainer.generative.gans.model.GAN'
gen_kls='src.explainer.generative.gans.generators.residual_generator.ResGenerator'
disc_kls='src.explainer.generative.gans.discriminators.simple_discriminator.SimpleDiscriminator'

#Check if the proto-model exist or build (empty):
get_dflts_to_of(local_config, 'proto_model',proto_kls)
proto_model_snippet = local_config['parameters']['proto_model']

#Check or set the default values for epoch and batch_size:
proto_model_snippet['parameters']['epochs'] = proto_model_snippet['parameters'].get('epochs', 100)
proto_model_snippet['parameters']['batch_size'] = proto_model_snippet['parameters'].get('batch_size', 8)

#Check if the generator exist or build with its defaults:
init_dflts_to_of(proto_model_snippet, 'generator', gen_kls, 7)
#Check if the generator exist or build with its defaults:
init_dflts_to_of(proto_model_snippet, 'discriminator', disc_kls, \
                    200,7)

#Check if the optimizer exist or build with its defaults:
init_dflts_to_of(proto_model_snippet, 'optimizer','torch.optim.Adam')
#Check if the loss function exist or build with its defaults:
init_dflts_to_of(proto_model_snippet, 'loss_fn','torch.nn.CrossEntropyLoss')

print(local_config)

