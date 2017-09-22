library(warbleR);
library(tuneR);
library(seewave);

setwd("~/USP/tcc/code/Rexp");

audio <- tuneR::readWave('100041.wav');
audio2 <- tuneR::readWave('100041.filtered1.wav');

#ad <- autodetec(threshold = 5, envt = "hil", ssmooth = 300, power=1,
#                bp=c(2,9), xl = 2, picsize = 2, res = 200, flim= c(1,11), osci = TRUE,
#              wl = 300, ls = FALSE, sxrow = 2, rows = 4, mindur = 0.1, maxdur = 1, set = TRUE, redo = TRUE, path = "~/USP/tcc/code/Rexp/")

#ad <- autodetec(threshold = 5, envt = "hil", ssmooth = 200, bp = c(2,9), mindur = 0.1, set = TRUE, redo = TRUE)

## Long Spectrogram
lspec(sxrow = 2, rows = 8, pal = reverse.heat.colors, wl = 300)

## Amplitude evenlope
env(audio, envt = "abs", msmooth = c(10, 50))


## Signal Detection
power <- 2
thres <- 20/100
wave1 <- env(audio, envt = "abs")
if (power != 1) wave1 <- wave1^power
wave2 <- ifelse(wave1 <= thres, yes = 1, no = 2)
n2 <- length(wave2)
wave4 <- apply(as.matrix(1:(n2-1)), 1, function(x) wave2[x] + wave2[x+1])
## wave4 := {2 = silence, 3 = change,  4 = signal}
n4 <- length(wave4)
wave4[c(1,n4)] <- 3
wave5 <- which(wave4 == 3)

#for (envts in c("abs", "hil")){
#  for (ssmo in c(100,200,300,400,500)) {
#    for (thrs in c(5, 10, 15, 20, 30, 50, 60, 70, 80, 90))
#    {
#      print(c(envts, ssmo, thrs));
#      ad <- autodetec(threshold = thrs, envt = envts, ssmooth = ssmo, bp = c(2,9), mindur = 0.1, set = TRUE, redo = TRUE);
#    }
#  }
#}