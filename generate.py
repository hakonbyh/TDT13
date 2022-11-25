from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

tokenizer = AutoTokenizer.from_pretrained("my_model/checkpoint-1000")
model = AutoModelForSeq2SeqLM.from_pretrained("my_model/checkpoint-1000")

inputs = "Prime minister | Norway | Jonas"

inputs = tokenizer(inputs, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]

print(predicted_title)
