import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
import time
# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxTTS.from_pretrained(device=device)

# Load the fine-tuned model
# model = ChatterboxTTS.from_local(
#     # "./checkpoints/chatterbox_finetuned_yodas",
#     "src/checkpoints/chatterbox_finetuned_t13n",  # change to your actual path
#     device=device
# )
# text = "Ezreal and Jinx teamed up with Ahri [giggle], Yasuo, and Teemo to take down [exhale] the enemy's Nexus in an epic late-game pentakill. [whistle]"
# text = "Ezreal and Jinx साउथ दिल्ली नगर निगम सख्त fucked up with lavda "
text = "toh, aapne jo gold loan liya hai, uska 15000 payment baaki hai, and its due on 10th july, to payment kab tak kar paoge aap??"
# text = "namaste, me meera bol rahi hu muthoot finance se, mene aapki home loan ke liye call kiya tha"
start_time = time.time()
wav = model.generate(text,exaggeration=0.4,
        cfg_weight=0.8,
        temperature=0.2,)
print(f"time taken: {time.time() - start_time}")
ta.save("c4.wav", wav, model.sr)