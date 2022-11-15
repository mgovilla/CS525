import matplotlib.pyplot as plt

with open('rewards.total', 'r') as f:
    line = f.readline()
    split = line[1:-2].split(', ')
    total, avgs = 0, []
    for i in range(len(split)):
        total += float(split[i])
        if i % 30 == 29:
            avgs.append(total / 30.0)
            total = 0


    plt.plot(avgs)
    plt.xlabel('epoch')
    plt.title('30 episode moving average of rewards')
    plt.ylabel('reward')
    plt.savefig("figure0.png")