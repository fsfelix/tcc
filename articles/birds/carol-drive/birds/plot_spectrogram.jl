function plot_spectrogram(s, fs)
    S = spectrogram(s[:,1], convert(Int, 25e-3*fs),
                    convert(Int, 10e-3*fs); window=hanning)
    t = time(S)
    f = freq(S)
    imshow(flipud(log10(power(S))), extent=[first(t), last(t),
             fs*first(f), fs*last(f)], aspect="auto")
    S
end