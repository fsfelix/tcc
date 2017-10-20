from multiprocessing import Pool, Lock, cpu_count, RLock

from functools import partial

def work(k, n):
    l = []
    for i in range(n):
        l.append((k, i))
    return l

def test(t):
    print(t[0])
    print(t[1])
    print(t[2])
    print(t[3])
    return t[0]
def teste():
    p = Pool(32)
    # ns = [[300000, 2],400000,500000,600000,700000,800000,900000,1000000,1100000,]
    ns = [(300,400,500,600,800),
          (30320,400,5002,30)]

    with open('results.txt', 'w') as f:
        for result in p.imap(test, ns):
            f.write(str(result))
            f.write('\n')

if __name__=='__main__':
    teste()
