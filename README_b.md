
TODO

- ZeRO3 unwrapping for faster generation (https://github.com/huggingface/trl/pull/1483)


## Tokenization with chat template

The best practice for tokenizing the chat-templated messages is to never go through a "text" only view even though many OSS libraries work that way (e.g., HF chat template).

```
# In this pseudo-code above, note that the tokenize method should not add a
# BOS or EOS token automatically, but should add a prefix space.
def tokenize(text):
    return tok.encode(text, add_special_tokens=False)

[BOS_ID] + tokenize("[INST]") + tokenize(USER_MESSAGE_1) + tokenize("[/INST]") + tokenize(BOT_MESSAGE_1) + [EOS_ID] + ... tokenize("[INST]") + tokenize(USER_MESSAGE_N) + tokenize("[/INST]") + tokenize(BOT_MESSAGE_N) + [EOS_ID]
```

Taking the text "<|im_start|>assistant\nHello" as example. In inference, we want to use tokens "<|im_start|>", "assistant", "\n" to predict "Hello". However, if we go through text only, and tokenize all "<|im_start|>assistant\nHello", we could get some unexpected tokens such as ">a" and "t\n", although this can be fixed by setting role/prefix tokens ("<|im_start|>", "assistant") as special tokens to prevent unexpected splits.

Then what about "<|im_start|>assistant Hello"? With the set special tokens, it will actually be tokenized into "<|im_start|>", "assistant", and " Hello", instead of "Hello"! We may have to set rstrip to "assistant" token or choose other symbols like "\n" instead of " ".

Moreover, some times we want to mask the question and the role/prefix tokens to caculating the loss only on output. In some implementaion, we first tokenize "<|im_start|>assistant Hello", then tokenize "<|im_start|>assistant " to get the number of tokens to mask. This may also trigger some errors.

Actually, if special token set, can just use "<|im_start|><assistant>Hello<im_end>"

The right thing to do is to always combine role/prefix token and text tokens as shown. However, sometimes we still tokenize chat-templated messages directly. Things we have to do is to make sure tokens in inference are same with the ones in training.

```python
# meta-llama/Meta-Llama-3-8B

s = "<|im_start|>\nHi"
print(tokenizer(s, add_special_tokens=False)["input_ids"])
# [128011, 198, 13347]
s = "<|im_start|>\n"
print(tokenizer(s, add_special_tokens=False)["input_ids"])
# [128011, 198]
s = "\nHi"
print(tokenizer(s, add_special_tokens=False)["input_ids"])
# [198, 13347]
s = "<|im_start|>"
print(tokenizer(s, add_special_tokens=False)["input_ids"])
# [128011]
s = "\n"
print(tokenizer(s, add_special_tokens=False)["input_ids"])
# [198]
s = "Hi"
print(tokenizer(s, add_special_tokens=False)["input_ids"])
# [13347]
```

```python
# meta-llama/Meta-Llama-3-8B

s = "<|im_start|> Hi"
print(tokenizer(s, add_special_tokens=False)["input_ids"])
# [128011, 21694]
s = "<|im_start|> "
print(tokenizer(s, add_special_tokens=False)["input_ids"])
# [128011, 220]
s = " "
print(tokenizer(s, add_special_tokens=False)["input_ids"])
# [220]
s = " Hi"
print(tokenizer(s, add_special_tokens=False)["input_ids"])
# [21694]
```

```python
# meta-llama/Meta-Llama-3-8B

def _process(s):
    input_ids = tokenizer(s, add_special_tokens=False)["input_ids"]
    input_string = tokenizer.convert_ids_to_tokens(input_ids)
    print(input_ids, input_string, sep=" ")
    return input_ids, input_string

s = "Hi"
_process(s)
# [13347] ['Hi']
s = " Hi"
_process(s)
# [21694] ['ĠHi']
s = "\nHi"
_process(s)
# [198, 13347] ['Ċ', 'Hi']
s = "\n1. Hi"
_process(s)
# [198, 16, 13, 21694] ['Ċ', '1', '.', 'ĠHi']
s = "1"
_process(s)
# [16] ['1']
```