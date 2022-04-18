import torch 

from wsol_model.ViT import vit_base_patch16_224, vit_large_patch16_224
from wsol_model.ViT import deit_base_patch16_224, deit_small_patch16_224

def vitol(architecture_type=None, pretrained=False, pretrained_path=None,**kwargs):
    print("Kwargs: ", kwargs)
    vit_type = kwargs['vit_type']

    if vit_type == 'vit':
        model = vit_base_patch16_224(pretrained=pretrained, num_classes=kwargs['num_classes'], adl_layer=kwargs['adl_layer'], adl_drop_rate=kwargs['adl_drop_rate'], adl_threshold=kwargs['adl_drop_threshold'])
    elif vit_type == 'vit_large':
        model = vit_large_patch16_224(pretrained=pretrained, num_classes=kwargs['num_classes'], adl_layer=kwargs['adl_layer'], adl_drop_rate=kwargs['adl_drop_rate'], adl_threshold=kwargs['adl_drop_threshold'])
    elif vit_type == 'vit_deit':
        model = deit_base_patch16_224(pretrained=pretrained, num_classes=kwargs['num_classes'], adl_layer=kwargs['adl_layer'], adl_drop_rate=kwargs['adl_drop_rate'], 
                        adl_threshold=kwargs['adl_drop_threshold'])
    elif vit_type == 'deit_small':
        model = deit_small_patch16_224(pretrained=pretrained, num_classes=kwargs['num_classes'], adl_layer=kwargs['adl_layer'], adl_drop_rate=kwargs['adl_drop_rate'], 
                        adl_threshold=kwargs['adl_drop_threshold'])
    return model

def generate_cam(attribution_generator, img, class_index, eval_method="lrp"):
    device = img.device 
    l = []
    B = img.size(0)
    for i in range(B):
        if eval_method == 'lrp':
            transformer_attribution = attribution_generator.generate_LRP(img[i].unsqueeze(0).to(device), method="transformer_attribution",index=class_index[i]).detach()
        elif eval_method == 'grad_rollout':
            transformer_attribution = attribution_generator.generate_grad_rollout(img[i].unsqueeze(0).to(device), index=class_index[i]).detach()
        elif eval_method == 'rollout':
            transformer_attribution = attribution_generator.generate_rollout(img[i].unsqueeze(0).to(device)).detach()

        transformer_attribution = transformer_attribution.reshape(1, 14, 14)
        l.append(transformer_attribution)
    return torch.cat(l, 0)