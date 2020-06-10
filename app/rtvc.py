def init():
  print("initializing rtvc engine")

def silence_removal(wav_fn):
  sox.transform()
  print(f"remove silence from {wav_fn}")

def convert(text, wav_fn):
  print(f"convert {text} using {wav_fn}")
  
