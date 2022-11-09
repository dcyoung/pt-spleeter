from typing import Dict
import numpy as np

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


def parse_int_or_default(s: str, default: int = 0) -> int:
    try:
        return int(s)
    except:
        return default


def tf2pytorch(checkpoint_path: str) -> Dict:
    init_vars = tf.train.list_variables(checkpoint_path)

    tf_vars = {}
    for name, _ in init_vars:
        try:
            # print('Loading TF Weight {} with shape {}'.format(name, shape))
            data = tf.train.load_variable(checkpoint_path, name)
            tf_vars[name] = data
        except Exception as e:
            print(f"Load error: {name}")
            raise

    layer_idxs = set(
        [
            parse_int_or_default(name.split("/")[0].split("_")[-1], default=0)
            for name in tf_vars.keys()
            if "conv2d_transpose" in name
        ]
    )

    n_layers_per_unet = 6
    n_layers_in_chkpt = max(layer_idxs) + 1
    assert (
        n_layers_in_chkpt % 6 == 0
    ), f"expected multiple of {n_layers_per_unet}... ie: {n_layers_per_unet} layers per unet & 1 unet per stem"
    n_stems = n_layers_in_chkpt // n_layers_per_unet

    stem_names = {
        2: ["vocals", "accompaniment"],
        4: ["vocals", "drums", "bass", "other"],
        5: ["vocals", "piano", "drums", "bass", "other"],
    }.get(n_stems, [])

    assert stem_names, f"Unsupported stem count: {n_stems}"

    state_dict = {}
    tf_idx_conv = 0
    tf_idx_tconv = 0
    tf_idx_bn = 0

    for stem_name in stem_names:
        # Encoder Blocks (Down sampling)
        for layer_idx in range(n_layers_per_unet):
            prefix = f"stems.{stem_name}.encoder_layers.{layer_idx}"
            conv_suffix = "" if tf_idx_conv == 0 else f"_{tf_idx_conv}"
            bn_suffix = "" if tf_idx_bn == 0 else f"_{tf_idx_bn}"

            state_dict[f"{prefix}.conv.weight"] = np.transpose(
                tf_vars[f"conv2d{conv_suffix}/kernel"], (3, 2, 0, 1)
            )
            state_dict[f"{prefix}.conv.bias"] = tf_vars[f"conv2d{conv_suffix}/bias"]
            tf_idx_conv += 1

            state_dict[f"{prefix}.bn.weight"] = tf_vars[
                f"batch_normalization{bn_suffix}/gamma"
            ]
            state_dict[f"{prefix}.bn.bias"] = tf_vars[
                f"batch_normalization{bn_suffix}/beta"
            ]
            state_dict[f"{prefix}.bn.running_mean"] = tf_vars[
                f"batch_normalization{bn_suffix}/moving_mean"
            ]
            state_dict[f"{prefix}.bn.running_var"] = tf_vars[
                f"batch_normalization{bn_suffix}/moving_variance"
            ]
            tf_idx_bn += 1

        # Decoder Blocks (Up sampling)
        for layer_idx in range(n_layers_per_unet):
            prefix = f"stems.{stem_name}.decoder_layers.{layer_idx}"
            tconv_suffix = "" if tf_idx_tconv == 0 else f"_{tf_idx_tconv}"
            bn_suffix = f"_{tf_idx_bn}"

            state_dict[f"{prefix}.tconv.weight"] = np.transpose(
                tf_vars[f"conv2d_transpose{tconv_suffix}/kernel"], (3, 2, 0, 1)
            )
            state_dict[f"{prefix}.tconv.bias"] = tf_vars[
                f"conv2d_transpose{tconv_suffix}/bias"
            ]
            tf_idx_tconv += 1

            state_dict[f"{prefix}.bn.weight"] = tf_vars[
                f"batch_normalization{bn_suffix}/gamma"
            ]
            state_dict[f"{prefix}.bn.bias"] = tf_vars[
                f"batch_normalization{bn_suffix}/beta"
            ]
            state_dict[f"{prefix}.bn.running_mean"] = tf_vars[
                f"batch_normalization{bn_suffix}/moving_mean"
            ]
            state_dict[f"{prefix}.bn.running_var"] = tf_vars[
                f"batch_normalization{bn_suffix}/moving_variance"
            ]
            tf_idx_bn += 1

        # Final conv2d
        state_dict[f"stems.{stem_name}.up_final.weight"] = np.transpose(
            tf_vars[f"conv2d_{tf_idx_conv}/kernel"], (3, 2, 0, 1)
        )
        state_dict[f"stems.{stem_name}.up_final.bias"] = tf_vars[
            f"conv2d_{tf_idx_conv}/bias"
        ]
        tf_idx_conv += 1

    return state_dict
