# Iterating through frames
# frame i, size N, hopsize H of signal x

function frame_i(x, i, N, H)
    size_x = size(x)[1];
    if i*N <= size_x
        return x[1 + (i - 1)*N : i*N];
    else
        return x[1 + (i - 1)*N : size_x];
    end
end


function all_frames(x, N, H)
    x_size = size(x)[1];
    numberOfFrames = ceil(Int, x_size/N);
    for i =1:numberOfFrames
        print(frame_i(x, i, N, H))
    end
end

function energyEnvelope(x, N, H)
    x_size = size(x)[1];
    numberOfFrames = ceil(Int, x_size/N);
    E = zeros(numberOfFrames)
    for i = 1:numberOfFrames
        frame_ = frame_i(x, i, N, H); 
        for j = 1:N
            #println(N);
            #println(size(frame_))
            if i < N
                E[i] += 20*log10(frame_[j]*frame_[j]);
            else
                ub = size(frame_)[1];
                if j <= ub
                    E[i] += 20*log10(frame_[j]*frame_[j]);
                end
            end
        end
    end
    return E;
end


for i = 1:NumberOfFrames
    println(frame_i(y, i, 128, 64))
end

for i = 1:NumberOfWindows
    if (i + 1)*N <= ysize
        println(y[i*N: (1 + i)*N])
    else
        println(y[i*N: ysize])
    end
end

function plot_spectrogram(s, fs)
    S = spectrogram(s[:,1]; window=hanning)
    t = time(S)
    f = freq(S)
    
    imshow(flipdim(log10(power(S)),1), extent=[first(t), last(t),
             fs*first(f), fs*last(f)], aspect="auto")
    S
end

plot_spectrogram(x, fs)

