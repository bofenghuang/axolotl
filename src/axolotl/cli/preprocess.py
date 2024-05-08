"""
CLI to run training on a model
"""
import logging
from pathlib import Path
from typing import Union

import fire
import transformers
from accelerate import init_empty_weights
from colorama import Fore
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM

from axolotl.cli import (
    check_accelerate_default_config,
    check_user_token,
    load_cfg,
    load_datasets,
    load_rl_datasets,
    print_axolotl_text_art,
)
from axolotl.common.cli import PreprocessCliArgs
from axolotl.common.const import DEFAULT_DATASET_PREPARED_PATH
from axolotl.prompt_strategies.sharegpt import (
    register_chatml_template,
    register_llama3_template,
    register_vigogne_chat_v4_template,
)

LOG = logging.getLogger("axolotl.cli.preprocess")


def do_cli(config: Union[Path, str] = Path("examples/"), **kwargs):
    # pylint: disable=duplicate-code
    print_axolotl_text_art()
    parsed_cfg = load_cfg(config, **kwargs)
    parsed_cfg.is_preprocess = True
    check_accelerate_default_config()
    check_user_token()
    parser = transformers.HfArgumentParser((PreprocessCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    if parsed_cfg.chat_template == "chatml":
        if parsed_cfg.default_system_message:
            LOG.info(
                f"ChatML set. Adding default system message: {parsed_cfg.default_system_message}"
            )
            register_chatml_template(parsed_cfg.default_system_message)
        else:
            register_chatml_template()
    elif parsed_cfg.chat_template == "llama3":
        if parsed_cfg.default_system_message:
            LOG.info(
                f"LLaMA-3 set. Adding default system message: {parsed_cfg.default_system_message}"
            )
            register_llama3_template(parsed_cfg.default_system_message)
        else:
            register_llama3_template()
    elif parsed_cfg.chat_template == "vigogne_chat_v4":
        register_vigogne_chat_v4_template()

    if parsed_cfg.chat_template == "llama3" and parsed_cfg.default_system_message:
        LOG.info(
            f"LLaMA-3 set. Adding default system message: {parsed_cfg.default_system_message}"
        )
        register_llama3_template(parsed_cfg.default_system_message)
    else:
        register_llama3_template()

    if parsed_cfg.chat_template == "vigogne_chat_v4":
        register_vigogne_chat_v4_template()

    if not parsed_cfg.dataset_prepared_path:
        msg = (
            Fore.RED
            + "preprocess CLI called without dataset_prepared_path set, "
            + f"using default path: {DEFAULT_DATASET_PREPARED_PATH}"
            + Fore.RESET
        )
        LOG.warning(msg)
        parsed_cfg.dataset_prepared_path = DEFAULT_DATASET_PREPARED_PATH

    if parsed_cfg.rl:  # and parsed_cfg.rl != "orpo":
        load_rl_datasets(cfg=parsed_cfg, cli_args=parsed_cli_args)
    else:
        load_datasets(cfg=parsed_cfg, cli_args=parsed_cli_args)

    if parsed_cli_args.download:
        model_name = parsed_cfg.base_model
        with init_empty_weights():
            AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    LOG.info(
        Fore.GREEN
        + f"Success! Preprocessed data path: `dataset_prepared_path: {parsed_cfg.dataset_prepared_path}`"
        + Fore.RESET
    )


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(do_cli)
