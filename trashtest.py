from src.utils.cfg_utils import init_dflts_to_of, get_dflts_to_of, set_if_not

local_config = {'parameters':{}}
proto_kls='src.explainer.generative.gans.model.GAN'
gen_kls='src.explainer.generative.gans.generators.residual_generator.ResGenerator'
disc_kls='src.explainer.generative.gans.discriminators.simple_discriminator.SimpleDiscriminator'

get_dflts_to_of(local_config, 'proto_model',proto_kls)
proto_model_snippet = local_config['parameters']['proto_model']

init_dflts_to_of(proto_model_snippet, 'generator', gen_kls, 8)
init_dflts_to_of(proto_model_snippet, 'discriminator', disc_kls, \
                  200,8)

set_if_not(proto_model_snippet, 'lr', {'generator': 0.001,'discriminator': 0.01})
            
init_dflts_to_of(local_config,'sampler','src.utils.n_samplers.partial_order_samplers.PositiveAndNegativeEdgeSampler')

#TODO: those are useles call if are equal to defaults of TorchBase
init_dflts_to_of(proto_model_snippet, 'optimizer','torch.optim.Adam') 
init_dflts_to_of(proto_model_snippet, 'loss_fn','torch.nn.CrossEntropyLoss')
print(local_config)

