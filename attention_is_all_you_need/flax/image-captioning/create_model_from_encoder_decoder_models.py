"""
Create a VisionEncoderDecoderModel instance from pretrained encoder/decoder models.

The cross-attention will be randomly initialized.
"""

from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer, FlaxVisionEncoderDecoderModel, HfArgumentParser


@dataclass
class ModelArguments:
    """
    Arguments pertaining to whic model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    output_dir: str= field(
        metadata={"help": "The output directory where the model will be written."},
    )
    encoder_model_name_or_path: str= field(
        metadata={
            "help": (
                "The encoder model checkpoint for weights initialization. "
                "Don't set if you want to train an encoder model from scratch."
            )
        },
    )
    decoder_model_name_or_path: str= field(
        metadata={
            "help": (
                "The decoder model checkpoint for weights initialization. "
                "Don't set if you want to train a decoder model from scratch."
            )
        },
    )
    encoder_config_name: Optional[str]= field(
        default=None, metadata={"help": "Pretrained encoder config name of path if not the same as encoder_model_name"}
    )
    decoder_config_name: Optional[str]= field(
        default=None, metadata={"help": "Pretrained decoder config name or path if not the same as decoder_model_name"}
    )


def main():
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_args_into_dataclasses()

    # Load pretrained model and tokenizer

    # Use explicit specified encoder config
    if model_args.encoder_config_name:
        encoder_config = AutoConfig.from_pretrained(model_args.encoder_config_name)
    # Use pretrained encoder model's config
    else:
        encoder_config = AutoConfig.from_pretrained(model_args.encoder_model_name_or_path)
    
    # Use explicit specified decoder config
    if model_args.decoder_config_name:
        decoder_config= AutoConfig.from_pretrained(model_args.decoder_config_name)
    # Use pretrained decoder model's config
    else:
        decoder_config= AutoConfig.from_pretrained(model_args.decoder_model_name_or_path)
    
    # necessary for 'from_decoder_pretrained' when 'decoder_config' is passed
    decoder_config.is_decoder= True
    decoder_config.add_cross_attention= True

    model= FlaxVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=model_args.encoder_model_name_or_path,
        decoder_pretrained_model_name_or_path=model_args.decoder_model_name_or_path,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )

    # GP2 only has bos/eos tokens but not decoder_start/pad tokens
    decoder_start_token_id= decoder_config.decoder_start_token_id
    pad_token_id= decoder_config.pad_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id= decoder_config.bos_token_id
    if pad_token_id is None:
        pad_token_id= decoder_config.eos_token_id
    
    # This is necessary to make Flax's generate() work
    model.config.eos_token_id= decoder_config.eos_token_id
    model.config.decoder_start_token_id= decoder_start_token_id
    model.config.pad_token_id= pad_token_id

    image_processor= AutoImageProcessor.from_pretrained(model_args.encoder_model_name_or_path)

    tokenizer= AutoTokenizer.from_pretrained(model_args.decoder_model_name_or_path)
    tokenizer.pad_token= tokenizer.convert_ids_to_tokens(model.config.pad_token_id)

    model.save_pretrained(model_args.output_dir)
    image_processor.save_pretrained(model_args.output_dir)
    tokenizer.save_pretrained(model_args.output_dir)


if __name__ == "__main__":
    main()