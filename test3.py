from jinja2.compiler import generate
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, pipeline

# # Załaduj dostrojony model
# model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")
# tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
#
# # Testowy prompt
# input_text = "diabetes"
# input_ids = tokenizer.encode(input_text, return_tensors='pt')
#
# # Generowanie tekstu
# output = model.generate(input_ids, max_length=100, num_return_sequences=1)
#
# # Dekodowanie tokenów do tekstu
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
prompt = 'Sugar for diabetics'
generator = pipeline('text-generation', model='./fine_tuned_model')
generated_text = generator(prompt, truncation=False, max_length=100, num_return_sequences=1)


print(generated_text)
