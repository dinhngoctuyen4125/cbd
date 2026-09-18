from .tofu import Deepseek_DataModule

def create_datamod(dataset_config, conv_template_config, data_mode_config, tokenizer=None, **kwargs):
    print(dataset_config)
    class_name = dataset_config.get('class_name', None)
    if "Deepseek".lower() in class_name.lower():
        mod = Deepseek_DataModule
    else:
        raise ValueError(f"Unknown data module class: {class_name}")

    return mod(
        tokenizer=tokenizer,
        conv_template_config=conv_template_config,
        **dataset_config,
        **data_mode_config,
        **kwargs,
    )
