def syllable_durations(markers, min_dur):
    N = len(markers)
    durations = []
    init = 0
    while (init < N):
        nxt = init + 1
        found = False
        while (nxt < N and markers[nxt] - markers[init] >= min_dur):
            print("Dentro do while: {} {}".format(init,nxt))
            nxt += 1
            found = True
        if nxt != N - 1 and found:
            print("DENTRO DO IF: {} {}".format(init,nxt))
            print("DENTRO DO IF: {} {}".format(markers[init],markers[nxt-1]))
            durations.append(markers[nxt - 2] - markers[init])
        init = nxt - 2
    return durations

print(syllable_durations([23,24,25,80,82,84], 2))
