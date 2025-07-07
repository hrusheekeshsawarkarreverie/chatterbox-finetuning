import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
import time
from transformers import PreTrainedTokenizerFast

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Define wrapper class
class HFTokenizerWrapper:
    def __init__(self, tokenizer, device="cpu"):
        self.tokenizer = tokenizer
        self.device = device

    def text_to_tokens(self, text):
        print("Used input_ids:", self.tokenizer(text, return_tensors="pt").input_ids[0].tolist())

        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
    
    def tokenize(self, text):
        """Tokenize text and return list of tokens"""
        return self.tokenizer.tokenize(text)
    
    def convert_tokens_to_ids(self, tokens):
        """Convert tokens to their corresponding IDs"""
        return self.tokenizer.convert_tokens_to_ids(tokens)


# model = ChatterboxTTS.from_pretrained(device=device)
# Load your Hindi-compatible tokenizer
# Load tokenizer
tokenizer_raw = PreTrainedTokenizerFast(
    tokenizer_file="src/checkpoints/chatterbox_finetuned_hi/tokenizer.json"
)
tokenizer = HFTokenizerWrapper(tokenizer_raw, device=device)
# Load the fine-tuned model
model = ChatterboxTTS.from_local(
    # "./checkpoints/chatterbox_finetuned_yodas",
    "src/checkpoints/chatterbox_finetuned_hi",  # change to your actual path
    device=device
)
model.tokenizer = tokenizer  # <- inject it manually if not already handled inside `from_local`

# text = "Ezreal and Jinx teamed up with Ahri [giggle], Yasuo, and Teemo to take down [exhale] the enemy's Nexus in an epic late-game pentakill. [whistle]"
text = " किन्तु आधुनिक पांडित्य, न सिर्फ़ एक ब्राह्मण रामानंद के, एक जुलाहे कबीर का गुरु होने से, बल्कि दोनों के समकालीन होने से भी, इनकार करता है"
# text = "toh, aapne jo gold loan liya hai, uska 15000 payment baaki hai, and its due on 10th july, to payment kab tak kar paoge aap??"
# text = "namaste, me meera bol rahi hu muthoot finance se, mene aapki home loan ke liye call kiya tha"
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)

print("Tokens:", tokens)
print("IDs:", ids)
start_time = time.time()
wav = model.generate(text,exaggeration=0.4,
        cfg_weight=0.8,
        temperature=0.2,)
print(f"time taken: {time.time() - start_time}")
ta.save("a1.wav", wav, model.sr)
print("Waveform shape:", wav.shape)
print("Duration (sec):", wav.shape[1] / model.sr)
