"""
Module for testing customized prompt tokenizers.

python -m unittest tests.test_prompt_tokenizers_b

"""

import logging
import unittest
from copy import deepcopy

# import pytest
from transformers import AutoTokenizer

from axolotl.prompt_strategies.sharegpt import (
    register_llama3_template,
    register_vigogne_chat_v4_template,
    ShareGPTPromptTokenizingStrategy,
)
from axolotl.prompters import ShareGPTPrompterV2


register_llama3_template()
register_vigogne_chat_v4_template()

LOG = logging.getLogger("axolotl")


test_data = {
    "single_turn_no_sys": {
        "conversations": [
            {"from": "human", "value": "abc"},
            {"from": "gpt", "value": "ipsum"},
        ]
    },
    "single_turn_sys": {
        "conversations": [
            {"from": "system", "value": "lorem"},
            {"from": "human", "value": "abc"},
            {"from": "gpt", "value": "ipsum"},
        ]
    },
    "multi_turn_sys": {
        "conversations": [
            {"from": "system", "value": "lorem"},
            {"from": "human", "value": "abc"},
            {"from": "gpt", "value": "ipsum"},
            {"from": "human", "value": "123"},
            {"from": "gpt", "value": "sit"},
        ]
    },
    # "multi_turn_no_sys": {
    #     "conversations": [
    #         {"from": "human", "value": "abc"},
    #         {"from": "gpt", "value": "ipsum"},
    #         {"from": "human", "value": "123"},
    #         {"from": "gpt", "value": "sit"},
    #     ]
    # },
}


def prompt_strat(conversation, tokenizer):
    "Helper function to create a prompt strategy for testing."
    prompter = ShareGPTPrompterV2(conversation=conversation)
    return ShareGPTPromptTokenizingStrategy(
        prompter,
        tokenizer,
        False,
        2048,
    )


class TestPromptTokenizationStrategiesLlama3(unittest.TestCase):
    """
    Test class for prompt tokenization strategies.
    """

    def setUp(self) -> None:
        # pylint: disable=duplicate-code
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        self._print_token_id = self.tokenizer.bos_token_id

    def test_sharegpt_llama_3(self):
        "Make sure the sharegpt/llama is tokenized and formatted correctly."
        strat = prompt_strat("llama-3", self.tokenizer)

        def tokenize(conv):
            result = strat.tokenize_prompt(deepcopy(conv))
            return result["input_ids"], result["labels"]

        def decode(ids):
            return strat.tokenizer.decode(ids)

        def decode_labels(ids):
            return decode(list(map(lambda x: self._print_token_id if x == -100 else x, ids)))

        # fmt: off
        # No system message, single-turn
        ns_input_ids, ns_labels = tokenize(test_data["single_turn_no_sys"])
        assert decode(ns_input_ids) == """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

abc<|eot_id|><|start_header_id|>assistant<|end_header_id|>

ipsum<|eot_id|><|end_of_text|>"""
        assert decode_labels(ns_labels) == """<|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|>ipsum<|eot_id|><|end_of_text|>"""
        # assert ns_input_ids == [1, 518, 25580, 29962, 25638, 518, 29914, 25580, 29962, 23421, 2]

        # System message, single-turn conversations
        st_input_ids, st_labels = tokenize(test_data["single_turn_sys"])
        assert decode(st_input_ids) == """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

lorem<|eot_id|><|start_header_id|>user<|end_header_id|>

abc<|eot_id|><|start_header_id|>assistant<|end_header_id|>

ipsum<|eot_id|><|end_of_text|>"""
        assert decode_labels(st_labels) == """<|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|>ipsum<|eot_id|><|end_of_text|>"""

        # System message, multi-turn conversations
        mt_input_ids, mt_labels = tokenize(test_data["multi_turn_sys"])
        assert decode(mt_input_ids) == """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

lorem<|eot_id|><|start_header_id|>user<|end_header_id|>

abc<|eot_id|><|start_header_id|>assistant<|end_header_id|>

ipsum<|eot_id|><|end_of_text|><|start_header_id|>user<|end_header_id|>

123<|eot_id|><|start_header_id|>assistant<|end_header_id|>

sit<|eot_id|><|end_of_text|>"""
        assert decode_labels(mt_labels) == """<|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|>ipsum<|eot_id|><|end_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|>sit<|eot_id|><|end_of_text|>"""
        # fmt: on


class TestPromptTokenizationStrategiesVigogneChatV4(unittest.TestCase):
    """
    Test class for prompt tokenization strategies.
    """

    def setUp(self) -> None:
        # pylint: disable=duplicate-code
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        self._print_token_id = self.tokenizer.bos_token_id

    def test_sharegpt_llama_3(self):
        "Make sure the sharegpt/llama is tokenized and formatted correctly."
        strat = prompt_strat("vigogne_chat_v4", self.tokenizer)

        def tokenize(conv):
            result = strat.tokenize_prompt(deepcopy(conv))
            return result["input_ids"], result["labels"]

        def decode(ids):
            return strat.tokenizer.decode(ids)

        def decode_labels(ids):
            return decode(list(map(lambda x: self._print_token_id if x == -100 else x, ids)))

        # fmt: off
        # No system message, single-turn
        ns_input_ids, ns_labels = tokenize(test_data["single_turn_no_sys"])
        assert decode(ns_input_ids) == """<|begin_of_text|><|user|>abc<|end_of_text|><|assistant|>ipsum<|end_of_text|>"""
        assert decode_labels(ns_labels) == """<|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|>ipsum<|end_of_text|>"""
        # assert ns_input_ids == [1, 518, 25580, 29962, 25638, 518, 29914, 25580, 29962, 23421, 2]

        # System message, single-turn conversations
        st_input_ids, st_labels = tokenize(test_data["single_turn_sys"])
        assert (
            decode(st_input_ids)
            == """<|begin_of_text|><|system|>lorem<|end_of_text|><|user|>abc<|end_of_text|><|assistant|>ipsum<|end_of_text|>"""
        )
        assert decode_labels(st_labels) == """<|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|>ipsum<|end_of_text|>"""

        # System message, multi-turn conversations
        mt_input_ids, mt_labels = tokenize(test_data["multi_turn_sys"])
        assert (
            decode(mt_input_ids)
            == """<|begin_of_text|><|system|>lorem<|end_of_text|><|user|>abc<|end_of_text|><|assistant|>ipsum<|end_of_text|><|user|>123<|end_of_text|><|assistant|>sit<|end_of_text|>"""
        )
        assert decode_labels(mt_labels) == """<|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|>ipsum<|end_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|><|begin_of_text|>sit<|end_of_text|>"""
        # fmt: on