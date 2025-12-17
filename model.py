from timm.models.eva import _create_eva

def Offset_Encoder(embed_dim=64,depth=2, global_pool='token', img_size=(256,128), PDR=0.05, **kwargs):
    model_args = dict(
        patch_size=16,
        dynamic_img_size=True,
        img_size=img_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=4,
        qkv_bias=False,
        init_values=1.0e-05, # layer-scale
        rope_type='dinov3',
        rope_temperature=100,
        rope_rotate_half=True,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        num_reg_tokens=0,
        use_fc_norm=False,
        norm_layer=partial(LayerNorm, eps=1e-5),
        global_pool=global_pool,
        num_classes=0,
        proj_drop_rate=PDR,
    )
    model = _create_eva('vit_small_patch16',pretrained=False, **dict(model_args, **kwargs))
    return model
