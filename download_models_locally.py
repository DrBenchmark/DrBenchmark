from transformers import AutoTokenizer, AutoModel


f_in = open("./models.txt","r")
models = f_in.read().split("\n")
f_in.close()

def save_locally(model_name):

    print(f">> {model_name}")
    local_path = f"./models/{model_name.lower().replace('/','_')}/"
    
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(local_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(local_path)

for m in models:
    save_locally(m)