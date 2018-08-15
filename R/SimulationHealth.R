
path = "~/Documents/ETH/Thesis/1plusX/Data/thesis/1plusx/Results/837817/Simulated/SimulationDetails.csv"
data <- read.csv(file=path, header=TRUE, sep=",")

data$ROC = data$TPR / data$FPR

a = subset(data, SimulationId == 0)
plot(a$FPR, a$TPR, type="l")
a = subset(data, SimulationId == 1)
plot(a$FPR, a$TPR, type="l")

ggplot(data, aes(x=FPR, y=TPR, colour=factor(SimulationId))) + 
  geom_line() +
  xlab("FPR") + ylab("TPR") +
  theme_bw() +
  expand_limits(y=c(0,1), x=c(0,1)) 


path = "~/Documents/ETH/Thesis/1plusX/Data/thesis/1plusx/Results/597165/Regression_Test.csv"
data <- read.csv(file=path, header=TRUE, sep=",")

