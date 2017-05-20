# Pkg.add("Wavelets")
# Pkg.add("AudioIO")
# Pkg.add("Winston")
# Pkg.add("WAV")
# Pkg.add("PyPlot")


using DSP
using Wavelets
#using AudioIO
#using Winston
using WAV
using PyPlot
using Gadfly
using DataFrames
using Distributions
using MixtureModels
using StatsFuns
#Pkg.build("MixtureModels")

#include("./plot_spectrogram.jl")

#####using AudioIO
# f = AudioIO.open("/home/medeirosc/Documents/birds/Limn_rect090.wav")
# data = read(f)
# close(f)
####end

##### opening signal
s, fs = wavread("/home/medeirosc/Documents/birds/Limn_rect090.wav")
fs

xs = [0:1/fs:((length(s[:,1])-1)/fs)]
length(xs)/fs # approx 23s

PyPlot.plot(xs, s[:,1])
xlabel("Time [s]")

##### first spectrogram

    S = spectrogram(s[:,1],round(Int64,50e-3*fs),round(Int64,15e-3*fs),window=hanning)#, round(Int64,500e-3*fs),round(Int64,100e-3*fs),
    t = time(S)
    f = freq(S)
    imshow(flipud(log10(power(S))), extent=[first(t), last(t),
             fs*first(f), fs*last(f)], aspect="auto")
  xlabel("Time [s]")
  #xticks([0:4:22]*fs)

SpowArray = reshape(log10(power(S)),length(power(S)),1)
Gadfly.plot(x=SpowArray, Geom.histogram)

######## distribution of power, sum of gaussians

distroPower = collect(hist(SpowArray,200))

valBin = collect(distroPower[1])
numBin = length(distroPower[1])
proBin = distroPower[2]
PyPlot.plot(proBin)


peakA = indmax(proBin)
valBin[peakA]
# find persistent change of derivative
befor = collect(peakA:-1:1)
drvBe = diff(proBin[befor])
findPos = find(drvBe .> 0)
i = 1;
while(abs(findPos[i] - findPos[i+1]) != 1 || abs(findPos[i+1] - findPos[i+2]) != 1)
  i = i + 1
  if i > length(findPos)-2
    break
  end
end
limInf = befor[findPos[i],:]
valBin[limInf]

after = collect(peakA:1:numBin-1)
drvBe = diff(proBin[after])
findPos = find(drvBe .> 0)
i = 1;
while(abs(findPos[i] - findPos[i+1]) != 1 || abs(findPos[i+1] - findPos[i+2]) != 1)
  i = i + 1
  if i > length(findPos)-1
    break
  end
end
limSup = after[findPos[i],:]
valBin[limSup]


aux = abs(valBin[limInf] - valBin[peakA]) - abs(valBin[limSup] - valBin[peakA])
stdDev = abs(valBin[limSup] - valBin[peakA])/2
media = valBin[peakA];
if isless(aux[1,1],0.0)
    stdDev = abs(valBin[limInf] - valBin[peakA])/2;
end
limInfDev = findmin(abs(valBin - ones(length(valBin),1)*(media-stdDev)))
limSupDev = findmin(abs(valBin - ones(length(valBin),1)*(media+stdDev)))

numBinA = sum(proBin[limInfDev[2]:limSupDev[2],1])
distroA = rand(Normal(media, stdDev[1,1]), numBinA)

distroA = rand(Normal(peakA, limSupDev[2]-peakA), numBinA)

distroV = rand(Normal(media, 1), numBinA)

Gadfly.plot(x=SpowArray, Geom.histogram)

Gadfly.plot(layer(x=SpowArray,Geom.histogram, Theme(default_color=color("orange"))),
     layer(x=distroV, Geom.histogram, Geom.point,Theme(default_color=color("purple"))))

PyPlot.plot(proBin)




distroPower

distroPowerA = collect(hist(distroA,200))
proBinA = [0;distroPowerA[2]]

valBin = collect(distroPower[1])
valBinA = collect(distroPowerA[1])

inCommon = intersect(valBin,valBinA)
inG = findin(valBin, inCommon)
inA = findin(valBinA, inCommon)
length(inG[i]:inG[end])
proBin = [0;proBin]

PyPlot.plot(proBin[inG])
PyPlot.plot(proBinA[inA])
# [proBin[inG] proBinA[inA]]
newDistro = zeros(size(inG))


newDistro = proBin;

j = 1
for i in 1:length(inG)
  newDistro[inG[i]] = proBin[inG[i]] - proBinA[inA[j]]
  j = j + 1
end

PyPlot.plot(newDistro)


numBin = length(distroPower[1])
proBin = distroPower[2]


distroA = rand(Normal(media, stdDev[1,1]), numBinA)























# r = MixtureModels.fit_fmm(MultivariateNormal{PDiagMat}, SpowArray, 3, fmm_em()) #MultivariateNormal{PDMat}
# r = fit_fmm(MultivariateNormal{PDMat},SpowArray, 3, fmm_em(maxiter=50, display=:iter))
# x = [1 2 3 4 5 6 7 8]
x = befor
pop!(x)
deleteat!(x,3)
# std(x)
# mean(x)+2*std(x)
# fdn1 = Distributions.fit(Normal, x);
# (cu,ca) = quantile(fdn1, [0.5 0.95]);

length(t)
freq(S)
pow = power(S)
relPow = find(pow .> 1e-5)
snr = length(relPow)/(length(t)*length(f))
# change window if less than 0.5


#prodF = zeros(length(f),1);
summF = zeros(length(f),1);
for i in 1:length(t)
 # prodF[i] = prod(pow[i,:]);
  summF[i] = sum(pow[i,:]);
end
find(summF)

highestRelFreq = (0.5*fs/length(f))*maximum(find(summF))
roundHf = round(highestRelFreq)

##### filter frequencies for reducing sample rate

responsetype = Lowpass(roundHf; fs=fs)
prototype = Butterworth(8)
low_filter = digitalfilter(responsetype, prototype)

# Let's take a look at the filter response now
w = 0:0.01:pi
H = freqz(low_filter, w)

plot(fs/2*w/pi, 20*log10(abs(H)))
xlabel("Frequency [Hz]")
ylabel("Gain [dB]")

# Filtering our signal with the filter
sf = filt(low_filter, s[:,1])
plot(sf)
length(sf)

##### use new signal and resample (downsample)
sD = resample(sf,roundHf*2/fs)

##### spectogram of simplified signal

# try to find windows in which t and f have similar lengths
    S = spectrogram(sD,round(Int64,30e-3*fs),round(Int64,5e-3*fs),window=hanning)#, round(Int64,500e-3*fs),round(Int64,100e-3*fs),
    t = time(S)
    f = freq(S)
    imshow(flipud(log10(power(S))), extent=[first(t), last(t),
             fs*first(f), fs*last(f)], aspect="auto")
  xlabel("Time [s]")
  #xticks([0:4:22]*fs)

length(t)
freq(S)
pow = power(S)
relPow = find(pow .> 1e-5)
snr = length(relPow)/(length(t)*length(f))
# change window if less than 0.5

#prodF = zeros(length(f),1);
summF = zeros(length(f),1);
summT = zeros(length(f),1);
for i in 1:length(t)
 # prodF[i] = prod(pow[i,:]);
  summF[i] = sum(pow[i,:]);
  summT[i] = sum(pow[:,i]);
end
relFreq = find(summF)
relTime = find(summT)

newFreq = freq(S)[relFreq]
newTime = t[relTime]

powCrop = pow[relFreq,relTime]

##### try replot significant areas, or shapening areas
    imshow(flipud(log10(powCrop)), extent=[first(newTime), last(newTime),
             fs*first(newFreq), fs*last(newFreq)], aspect="auto")
# normalize powCrop
powN = zeros(size(powCrop))
up = maximum(powCrop)
dw = minimum(powCrop)
for i in 1:length(powCrop[:,1])
  for j in 1:length(powCrop[1,:])
    powN[i,j] = (powCrop[i,j]-dw)/(up-dw);
  end
end
  imshow(flipud(log10(powN)), extent=[first(newTime), last(newTime),
               fs*first(newFreq), fs*last(newFreq)], aspect="auto")

minimum(powN)
maximum(powN)


fdn1 = Distributions.fit(Normal,powNarray[:,1]);

powNarray = reshape(powN,length(powN[1,:])^2,1)
Gadfly.plot(x=log10(powNarray), Geom.histogram)


###======================================================here

# sharpening the blubs
sharp = [0.5 -1 0.5;-1 1 -1;0.5 -1 0.5];
powS = zeros(size(powN))
for i in 2:2:length(powN[:,1])-1
  for j in 2:2:length(powN[1,:])-1
    powS[i-1:i+1,j-1:j+1] = powN[i-1:i+1,j-1:j+1]*sharp;
  end
end

powSN = zeros(size(powCrop))
up = maximum(powS)
dw = minimum(powS)
for i in 1:length(powS[:,1])
  for j in 1:length(powS[1,:])
    powSN[i,j] = (powS[i,j]-dw)/(up-dw);
  end
end

up = maximum(powSN)
dw = minimum(powSN)

  imshow(flipud(log10(powSN)), extent=[first(newTime), last(newTime),
               fs*first(newFreq), fs*last(newFreq)], aspect="auto")







powM = zeros(size(pow))
up = maximum(powS)
dw = minimum(powS)
for j in 2:2:length(t)-1
  for i in 2:2:length(f)-1
    powM[i,j] = (powS[i,j]-dw)/(up-dw);
  end
end

    imshow(flipud(log10(powN)), extent=[first(newTime), last(newTime),
             fs*first(newFreq), fs*last(newFreq)], aspect="auto")
    imshow(flipud(powM), extent=[first(newTime), last(newTime),
             fs*first(newFreq), fs*last(newFreq)], aspect="auto")

# using Gadfly
# Gadfly.spy(powS)

maximum(powN)


# past implementation

arr = Array{length(s[:,1]),1};
arr = s[:,1]
i=1
for i in 1:length(s[:,1]) arr[i] = s[i,1] end
fftfreq(Array{:,1}(s[:,1]),44100)


length(s[:,1])
power(S)


responsetype = Bandpass(1000, 10000; fs=fs)
prototype = Butterworth(8)
info_filter = digitalfilter(responsetype, prototype)

# Let's take a look at the filter response now
w = 0:0.01:pi
H = freqz(info_filter, w)

plot(fs/2*w/pi, 20*log10(abs(H)))
xlabel("Frequency [Hz]")
ylabel("Gain [dB]")

# Filtering our signal with the filter
sf = filt(info_filter, s)

    S = spectrogram(sf[:,1],round(Int64,1e-3*fs),round(Int64,0.1e-3*fs), window=hanning)#, convert(Int, 25e-3*fs),convert(Int, 10e-3*fs)
    t = time(S)
    f = freq(S)
    imshow(flipud(log10(power(S))), extent=[first(t), last(t),
             fs*first(f), fs*last(f)], aspect="auto")
  xlabel("Time [s]")
  xticks([0:4:22]*fs)

#play(sf)
plot(xs, sf)


wt = wavelet(WT.db4)
x = sf[:,1]#sin(4*linspace(0,2*pi-eps(),1024))
tree = bestbasistree(x, wt)
xtb = wpt(x, wt, tree)
xt = dwt(x, wt)
# get biggest m-term approximations
m = 50
threshold!(xtb, BiggestTH(), m)
threshold!(xt, BiggestTH(), m)
# compare sparse approximations in ell_2 norm
vecnorm(x - iwpt(xtb, wt, tree), 2) # best basis wpt
vecnorm(x - idwt(xt, wt), 2)

wt = wavelet(WT.db4)
x = sf[1:10000,1]
ta = dwt(x, wt)
plot(ta)
plot(x)
tb = idwt(ta, wt,4)
plot(tb)
vecnorm(x - idwt(ta, wt), 2)
plot(ta - idwt(ta, wt))

# find frequency of each level





Pkg.update()
