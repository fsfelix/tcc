import numpy as np
# def syllable_durations(markers, min_dur):
#     N = len(markers)
#     durations = []
#     init = 0
#     while (init < N):
#         nxt = init + 1
#         found = False
#         while (nxt < N and markers[nxt] - markers[init] >= min_dur):
#             print("Dentro do while: {} {}".format(init,nxt))
#             nxt += 1
#             found = True
#         if nxt != N - 1 and found:
#             print("DENTRO DO IF: {} {}".format(init,nxt))
#             print("DENTRO DO IF: {} {}".format(markers[init],markers[nxt-1]))
#             durations.append(markers[nxt - 2] - markers[init])
#         init = nxt - 2
#     return durations

def syllable_durations(markers, max_dur):
    # markers.append(1000000)
    markers.append(np.inf)
    N = len(markers)
    print("N: {}".format(N))
    durations = []
    init = 0
    while (init < N):
        end = init + 1
        # print("first init: {} end: {}".format(init, end))
        while (end < N):
            if (markers[end] - markers[init] > max_dur):
                if (init != end - 1):
                    durations.append(markers[end - 1] - markers[init])
                # init = end
                print("if init: {} end: {}".format(init, end))
                if end + 1 == N:
                    print("era pra ter acabado")
                    init = N + 1
                else:
                    init = end
                break
            end += 1
        #print(durations)
    return durations

#print(syllable_durations([23,24,25,80,82,84], 2))
print(syllable_durations([0, 3, 4, 23, 24, 25, 1453, 1456, 1458, 1467, 8001, 8009, 8030], 15))
