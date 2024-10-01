from transformers import pipeline


prompt = 'Sugar for diabetics'
generator = pipeline('text-generation', model='./fine_tuned_model')
generated_text = generator(prompt, truncation=False, max_length=100, num_return_sequences=1)


print(generated_text)
