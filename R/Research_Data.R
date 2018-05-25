data <- read.csv(file="~/Documents/ETH/Thesis/1plusX/Data/Code/Click_Rates.csv", header=TRUE, sep=",")
data$PercentageClicks = data$Clicks / data$Impressions
totalImpressions = 2340996
totalClicks = sum(data$Clicks)
totalPercentageClicks = totalClicks/totalImpressions

plot(data$Impressions, data$PercentageClicks)
abline(h=rep(totalPercentageClicks, nrow(data)))

data.random <- read.csv(file="~/Documents/ETH/Thesis/1plusX/Data/R6/ydata-fp-td-clicks-v1_0.20090501_random_baseline.csv", header=TRUE, sep=",")
data.random$PercentageClicks = data.random$Clicks / data.random$Impressions
data.random.totalImpressions = 109979
data.random.totalClicks = sum(data.random$Clicks)
data.random.totalPercentageClicks = data.random.totalClicks/data.random.totalImpressions

plot(data.random$Impressions, data.random$PercentageClicks)
abline(h=rep(data.random.totalPercentageClicks, nrow(data.random)))


data.random.ctr <- read.csv(file="~/Documents/ETH/Thesis/1plusX/Data/R6/20090501_random_ctr_timeline.csv", header=TRUE, sep=",")
data.random.ctr$PercentageClicks = data.random.ctr$Clicks / data.random.ctr$Impressions
plot(data.random.ctr$PercentageClicks, type="l")

data.all.ctr <- read.csv(file="~/Documents/ETH/Thesis/1plusX/Data/R6/20090501_all_ctr_timeline.csv", header=TRUE, sep=",")
data.all.ctr$PercentageClicks = data.all.ctr$Clicks / data.all.ctr$Impressions
plot(data.all.ctr$PercentageClicks, type="l")

data.ucb.ctr <- read.csv(file="~/Documents/ETH/Thesis/1plusX/Data/R6/20090501_ucb_ctr_timeline.csv", header=TRUE, sep=",")
data.ucb.ctr$PercentageClicks = data.ucb.ctr$Clicks / data.ucb.ctr$Impressions
plot(data.ucb.ctr$Impressions, data.ucb.ctr$PercentageClicks, type="l")


data.linucb.lin.ctr <- read.csv(file="~/Documents/ETH/Thesis/1plusX/Data/R6/20090501_LibUCB_Lin_ctr_timeline.csv", header=TRUE, sep=",")
data.linucb.lin.ctr$PercentageClicks = data.linucb.lin.ctr$Clicks / data.linucb.lin.ctr$Impressions
plot(data.linucb.lin.ctr$Impressions, data.linucb.lin.ctr$PercentageClicks, type="l")

data.linucb.hybrid.ctr <- read.csv(file="~/Documents/ETH/Thesis/1plusX/Data/R6/20090501_LibUCB_Hybrid_ctr_timeline.csv", header=TRUE, sep=",")
data.linucb.hybrid.ctr$PercentageClicks = data.linucb.hybrid.ctr$Clicks / data.linucb.hybrid.ctr$Impressions
plot(data.linucb.hybrid.ctr$Impressions, data.linucb.hybrid.ctr$PercentageClicks, type="l")
lines(data.linucb.lin.ctr$Impressions)



