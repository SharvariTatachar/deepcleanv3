import matplotlib.pyplot as plt 
import json 

def load_run(path): 
    with open(path) as f: 
        return json.load(f)

hybrid = load_run('train_dir/perchannel_run1.json')
deepclean = load_run('train_dir/dc_run1.json')

plt.plot(hybrid["history"]["train_loss"], label="Hybrid train")
plt.plot(hybrid["history"]["val_loss"], "--", label="Hybrid val")

plt.plot(deepclean["history"]["train_loss"], label="DeepClean train")
plt.plot(deepclean["history"]["val_loss"], "--", label="DeepClean val")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("comparison.png")