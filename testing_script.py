from main_pyg import main
import os
import math

all_datasets = ["ogbg-molbace", "ogbg-moltox21", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast"]
datasets = ["ogbg-molbace"]
results = ""
for d in datasets:
    sum = 0
    values = list()
    for i in range(10):
        filename = "output" + str(i) + ".txt"
        os.system("python main_pyg.py --gnn gin --dataset " + d + " --filename alo --emb_dim 64 --epochs 100 --batch_size 32 >> "+filename)

        with open(filename) as f:
            lines = f.readlines()
            acc = float(lines[-1].split(' ')[-1])
            sum += acc
            values.append(acc)

    std = 0
    for i in values:
        std += (i-sum/10)*(i-sum/10)
    std /= 10
    std = math.sqrt(std)
    results += str(d + ": " + str(sum/10) + "  +-  " + str(std) + "\n")

print(results)
