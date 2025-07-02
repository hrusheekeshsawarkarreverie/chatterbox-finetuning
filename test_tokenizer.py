# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_local("./checkpoints/chatterbox_finetuned_hi")

# print(len(tokenizer))  # Should be 2000

# print(tokenizer.tokenize("साउथ दिल्ली नगर निगम सख्त"))
# # Should return: ['स', 'ा', 'उ', 'थ', ' ', 'द', 'ि', ...] or other valid Hindi subwords/chars

from transformers import AutoTokenizer

# Absolute path or relative path both work
dir_name="/teamspace/studios/this_studio/chatterbox-finetuning/checkpoints/chatterbox_finetuned_hi"
# tokenizer.save_pretrained(dir_name)

# tokenizer = AutoTokenizer.from_pretrained("/teamspace/studios/this_studio/chatterbox-finetuning/checkpoints/chatterbox_finetuned_hi",local_files_only=True,use_auth_token=False)

from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(tokenizer_file="/teamspace/studios/this_studio/chatterbox-finetuning/src/checkpoints/chatterbox_finetuned_hi/tokenizer.json")

text = "साउथ दिल्ली नगर निगम सख्त"
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)

print("Tokens:", tokens)
print("IDs:", ids)
