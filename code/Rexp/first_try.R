library(warbleR);
library(tuneR);

setwd("~/USP/tcc/code/Rexp")

#audio <- tuneR::readWave('100056.mp3.filtered1.wav');

ad <- autodetec(threshold = 5, envt = "hil", ssmooth = 300, power=1,
                bp=c(2,9), xl = 2, picsize = 2, res = 200, flim= c(1,11), osci = TRUE,
              wl = 300, ls = FALSE, sxrow = 2, rows = 4, mindur = 0.1, maxdur = 1, set = TRUE, redo = TRUE)