import subprocess

f_in = open("./models.txt","r")
models = f_in.read().split("\n")
f_in.close()

def save_locally(model_name):

    print(f">> {model_name}")
    local_path = f"./models/{model_name.lower().replace('/','_')}/"
    cmd = f"git clone https://huggingface.co/{model_name} {local_path}"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output, error)

for m in models:
    save_locally(m)