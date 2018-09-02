
path = "~/Documents/ETH/Thesis/1plusX/Data/thesis/1plusx/Results/837817/Simulated/SimulationDetails.csv"
data <- read.csv(file=path, header=TRUE, sep=",")
data$ROC = data$TPR / data$FPR

# When selecting the right simulation type we can use few metrics to evaluate it.
# 1) FPR vs TPR (it would be best if the line is above the (0,0) - (1,1) line)
# 2) ROC = TPR / FPR
# 3) NE = normalized enthropy - we want to be learning more.

ggplot(data, aes(x=FPR, y=TPR, colour=factor(SimulationId))) + 
  geom_line() +
  xlab("False Positive Rate") + ylab("True Positive Rate") + ggtitle("TPR vs FPR") +
  theme_bw() +
  expand_limits(y=c(0,1), x=c(0,1)) +
  scale_color_discrete(name="SimulationId")

ggplot(data, aes(x=Calibration, y=ROC, colour=factor(SimulationId))) + 
  geom_line() +
  xlab("Calibration") + ylab("ROC") + ggtitle("ROC = TPR/FPR") +
  theme_bw() +
  expand_limits(y=c(0,1), x=c(0,1)) +
  scale_color_discrete(name="SimulationId")

ggplot(data, aes(x=Calibration, y=NE, colour=factor(SimulationId))) + 
  geom_line() +
  xlab("Calibration") + ylab("Normalized Enthropy") + ggtitle("Normalized Enthropy") +
  theme_bw() +
  expand_limits(y=c(0,1), x=c(0,1)) +
  scale_color_discrete(name="SimulationId")

#library(gsubfn)
height = (data$TPR[-1]+data$TPR[-length(data$TPR)])/2
width = -diff(data$FPR) # = diff(rev(omspec))
sum(height*width)

data$SimulationId = factor(data$SimulationId)
height <- with(data, by(data, data$SimulationId, function(x) (x$TPR[-1]+x$TPR[-length(x$TPR)])/2))
width <- with(data, by(data, data$SimulationId, function(x) (diff(x$FPR))))

res <- data.frame(SimulationId=integer(), AUC=double())
for (i in seq(1,  length(height), 1)) {
  cur = sum(unlist(height[i]) * unlist(width[i]))
  cur_df <- data.frame(SimulationId=c(i), AUC=cur)
  res <- rbind(res, cur_df)
}

ggplot(res, aes(x=SimulationId, y=AUC)) + 
  geom_point() +
  xlab("SimulationId") + ylab("AUC") + ggtitle("AUC per Simulation") +
  theme_bw() +
  expand_limits(y=c(0,1)) 

