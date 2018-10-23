library(ggplot2)
library(scales)
library(gridExtra)

formatterM <- function(){
  function(x) paste(x/1000000, "M")
}
formatterPercent <- function(){
  function(x) paste(x, "%")
}
selectNecessaryColumns <- function(df, columns) {
  for (column in columns) {
    if(!(column %in% colnames(df))){
      df[column] = NA
    } 
  }
  df[,columns]
}

renameCampaignIds <-function (data) {
  campaignIds = c(847460, 856805, 858140, 865041, 866128)
  campaignPseudonims = c("A", "B", "C", "D", "E")
  
  index = 1
  for (campaignId in campaignIds)
  {
    data[data$CampaignId == campaignId, "CampaignId"] =  campaignPseudonims[index]
    index = index + 1
  }
  data
}

formatterPretty <- function(){
  function(x) ifelse(x==10, paste(x, "% (Random)"), paste(x, "%"))
}

formatterK <- function(){
  function(x) ifelse(x==0, "0", paste(x/1000, "K"))
}

formatterTime <- function(){
  function(x) as.POSIXct( x /1000, origin="1970-01-01")
}

formatterTimeInMin <- function(){
  function(x) paste(as.integer(x/(60)))
}

formatterTimeInHour <- function(){
  function(x) paste(as.integer(x/(60*60)))
}

formatterLog <- function(){
  function(x) ifelse(log(x) < 0, 0, round(log(x), digits=1))
}
formatterDensity <- function(){
  function(x) x*10000
}

getClickersPlot <-function(includeRealData = TRUE){
  data = data.frame(x=double(), type=integer())
  chi_dfs = c(1, 1, 10, 100, 100)
  chi_alphas = c( 1, 4, 4, 1, 4)
  for (index in 1:length(chi_dfs))
  {
    df = chi_dfs[index]
    alpha = chi_alphas[index]
    
    x =  rchisq(1000, df=df, ncp = 0) 
    y = alpha*(x - min(x)) / (max(x) - min(x))
    
    data1 = data.frame(x = y, type=paste("DF:",df, " M:", alpha, sep = ""))
    data = rbind(data, data1)  
  }
  if(includeRealData)
  {
    data1 = data.frame(x = rep(0, 1), type="Real Data")
    data = rbind(data, data1)  
  }
  
  p = ggplot(data, aes(x=x, color=factor(type), fill=factor(type))) +
    geom_density(aes(y = ..density..), size=1, alpha = 0.3) + 
    xlim(c(0, 1)) +
    xlab("Value") +
    ylab("Density") +
    scale_color_discrete(name="Chi Parameters") +
    scale_fill_discrete(guide=FALSE) +
    theme_bw() +
    ggtitle("Simulated Clickers Distribution (Chi)") +
    theme(plot.title = element_text(size = 10, face = "bold"), 
          legend.title=element_text(size=12), 
          legend.text=element_text(size=8), 
          legend.key.size = unit(1,"line"),
          axis.title.x = element_text(size=8),
          axis.title.y = element_text(size=8),
          legend.position="bottom") 
  p
}
prepData <- function(files, cur_path) {
  file=paste0(cur_path, paste0(files[1], ".csv"))
  data <- read.csv(file=paste0(cur_path, paste0(files[1], ".csv")), header=TRUE, sep=",")
  columns = c("Clicks", "Impressions", "RecommendationPart","TotalImpressions","Method", "Alpha","Timestamp","TrainPart","BatchCTR","ModelCTR","MSE","MMSE","FullMSE","FullROC","FullTPR","FullFPR","FullFNR","FullPPR","ModelCalibration","ModelNE","ModelRIG"
              , "Nu", "Hours", "LengthScale", "ClusterCount","EqClicks","LearningRate", "MSE", "TargetPercent", "TargetAlpha", "TargetSplit", "EarlyUpdate", "BatchMSE", "BatchMMSE", "FullMSE", "FullMMSE", "SimulationType",
              "CropPercent","NormalizeTargetValue","SimulationIndex","ChiDF", "ChiAlpha","SimulationId","CTRMultiplier")
  data = selectNecessaryColumns(data, columns)
  if(length(files) > 1){
    i = 1
    for (name in files){
      if(i != 1)
      {
        data1 <- read.csv(file=paste0(cur_path, paste0(name, ".csv")), header=TRUE, sep=",")
        data1 = selectNecessaryColumns(data1, columns)
        data <- rbind(data, data1)  
      }
      i= i + 1
    }
  }
  data$Percent = data$Clicks / data$Impressions * 100
  data$Factor = factor(paste(data$Method, " ", data$Alpha, " R:", data$RecommendationPart * 100, sep=""))
  data$UserPercentage = factor(data$RecommendationPart * 100)
  data$Timestamp = as.POSIXct(data$Timestamp, origin="1970-01-01")
  data
}
g_legend<-function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)}

plotGroup <- function(g1, g2, title, width = 8) {
  mylegend<-g_legend(g2)
  grid.arrange(arrangeGrob(g1 + theme(legend.position="none"),
                           g2 + theme(legend.position="none"),
                           nrow=1, ncol=2, widths=unit(c(width, width), "cm"), heights=unit(8, "cm")),
               mylegend, nrow=2, widths=unit(width*2, "cm"), heights=unit(c(8, 1.5), "cm"),
               top = textGrob(title, gp=gpar(fontsize=15,fontface=2)))
}

library(grid)
plotGroupNoLegend <- function(g1, g2, title, width) {
  grid.arrange(g1 + theme(legend.position="none"),
               g2 + theme(legend.position="none"),
               nrow=1, ncol=2, widths=unit(c(width, width), "cm"), heights=unit(9, "cm"),
               top = textGrob(title, gp=gpar(fontsize=15,fontface=2)))
  
}

plotGroupNoTitle <- function(g1, g2, width) {
  g2  = g2 + theme(legend.position="bottom")
  mylegend<-g_legend(g2)
  grid.arrange(arrangeGrob(g1 + theme(legend.position="none"),
                           g2 + theme(legend.position="none"),
                           nrow=1, ncol=2, widths=unit(c(width, width), "cm"), heights=unit(8.5, "cm")),
               mylegend, nrow=2, widths=unit(width*2, "cm"), heights=unit(c(8.5, 1.5), "cm"))
  
}

plotGroupNoTitleNoLegend <- function(g1, g2, width) {
  grid.arrange(g1 + theme(legend.position="none"),
               g2 + theme(legend.position="none"),
               nrow=1, ncol=2, widths=unit(c(width, width), "cm"), heights=unit(9, "cm"))
}

getCampaignComparisonPlot <-function (data, ylable, title, limit) {
  p1 = ggplot() + 
    geom_bar(aes(y = Ratio, x = factor(CampaignId), fill = factor(CampaignId)), data = data, stat="identity") + 
    xlab("Campaign Id") +
    ylab(ylable) +
    scale_fill_discrete(name="CampaignId") +
    theme_bw() +
    ggtitle(title) +
    expand_limits(y=c(0, limit)) +
    theme(#axis.text.x = element_text(angle = 45, hjust = 1),
          axis.title.x=element_blank(),
          plot.title = element_text(size=12)) +
    theme(legend.justification=c(1,0),legend.position="bottom") 
  p1
}

plotThreeComparison <-function (p1, p2, p3, title, saveImage, imagePath) {
  plot_count = 3
  plot_width = 5.5
  p5 = grid.arrange(arrangeGrob(p1 + theme(legend.position="none"),
                                p2 + theme(legend.position="none", axis.title.y=element_blank()),
                                p3 + theme(legend.position="none", axis.title.y=element_blank()),
                                nrow=1, ncol=plot_count, widths=unit(rep(plot_width, plot_count),"cm"), heights=unit(10, "cm")),
                    nrow=1, widths=unit(plot_count * plot_width, "cm"), heights=unit(10, "cm"),
                    top = textGrob(title, gp=gpar(fontsize=15, fontface=2)))
  if(saveImage){
      ggsave(imagePath, plot=p5, width = plot_width*plot_count, height = 10, units = "cm")  
  }
  
}

plotThreeComparisonNoTitle <-function (p1, p2, p3, saveImage, imagePath) {
  plot_count = 3
  plot_width = 5.5
  p5 = grid.arrange(arrangeGrob(p1 + theme(legend.position="none"),
                                p2 + theme(legend.position="none", axis.title.y=element_blank()),
                                p3 + theme(legend.position="none", axis.title.y=element_blank()),
                                nrow=1, ncol=plot_count, widths=unit(rep(plot_width, plot_count),"cm"), heights=unit(10, "cm")),
                    nrow=1, widths=unit(plot_count * plot_width, "cm"), heights=unit(10, "cm"))
  if(saveImage){
    ggsave(imagePath, plot=p5, width = plot_width*plot_count, height = 10, units = "cm")  
  }
  
}
getCTRPlot <- function(data, limits = c(0,1.5), scaleName="", pos="bottom") {
  g1 = ggplot(data, aes(x=TotalImpressions, y=Percent, colour=Factor)) + 
    geom_line(size=1) +
    xlab("Time Progression") + ylab("Cumulative CTR (Scaled)") +
    expand_limits(y=limits) + 
    #scale_x_continuous(labels=formatterM()) + 
    scale_y_continuous(labels=formatterPercent()) +
    scale_color_discrete(name=scaleName) +
    theme_bw() +
    theme(plot.title = element_text(size = 10, face = "bold"), 
          legend.title=element_text(size=12), 
          legend.text=element_text(size=8), 
          legend.key.size = unit(1,"line"),
          axis.title.x = element_text(size=8),
          axis.title.y = element_text(size=8),
          legend.position=pos) 
  
  g1
}

getMSEPlot <- function(data, xLimit, scaleName, pos = "bottom") {
  g2 = ggplot(data, aes(x=TotalImpressions, y=MSE, colour=Factor)) + 
    stat_smooth(aes(y=MSE, colour=Factor), method = "lm", formula = y ~ poly(x, 20), se = FALSE, size=1) +
    xlab("Time Progression") + ylab("Log MSE") +
    expand_limits(y=xLimit) + 
    scale_y_log10()+
   # scale_x_continuous(labels=formatterM()) +
    theme_bw() +
    theme(plot.title = element_text(size = 10, face = "bold"), 
          legend.title=element_text(size=12), 
          legend.text=element_text(size=8), 
          legend.position=pos,
          legend.key.size = unit(1,"line"),
          axis.title.x = element_text(size=8),
          axis.title.y = element_text(size=8))+
    scale_color_discrete(name=scaleName)
  g2
}

getMSEPlotLog <- function(data, xLimit, scaleName, pos = "bottom") {
  g2 = getMSEPlot(data, xLimit, scaleName, pos) +  scale_y_log10() + ylab("Log MSE")
  g2
}



getCampaignMetricsSimulated <- function(files, campaignId, baseCTR, method = "LinUCB_Disjoint", alpha = 0) {
  path = paste("~/Documents/ETH/Thesis/1plusX/Data/Thesis/1plusx/Results/", campaignId, "/Simulated/", sep="")
  
  getCampaignMetricsBase(path, files, campaignId, baseCTR, method, alpha)
}

getCampaignMetrics <- function(files, campaignId, baseCTR, method = "LinUCB_Disjoint", alpha = 0) {
  path = paste("~/Documents/ETH/Thesis/1plusX/Data/Thesis/1plusx/Results/", campaignId, "/", sep="")
  getCampaignMetricsBase(path, files, campaignId, baseCTR, method, alpha)
}

getCampaignMetricsBase <- function(path, files, campaignId, baseCTR, method = "LinUCB_Disjoint", alpha = 0) {
  data = prepData(files, path)
  
  data = subset(data, Timestamp < 1534709673000)
  
  if (method != "Random")
  {
    campaignData = subset(data, Impressions > 1000 & Alpha == alpha & (EqClicks == 0.0 | is.na(EqClicks)) & Method == method)  
  }
  else 
  {
    campaignData = data
  }
  maxTotalImpressions = max(campaignData$TotalImpressions)
  campaignData$CampaignId = campaignId
  campaignData$BaseCTR = baseCTR
  campaignData$CTR = campaignData$Clicks / campaignData$Impressions
  campaignData$Ratio = campaignData$CTR/campaignData$BaseCTR
  campaignData$RecommendationPart = campaignData$RecommendationPart * 100
  
  campaignData = campaignData[campaignData$TotalImpressions == maxTotalImpressions, 
                              c("CampaignId", "Method", "RecommendationPart", "Clicks", 
                                "Impressions", "FullFNR","FullTPR", "BaseCTR", "Ratio", "CTR", "Timestamp")]
  
  campaignData
}

getBarPlot <- function(data, yLabel, title){
  ggplot(data, aes(x=RecommendationPart, y=mean, fill=RecommendationPart)) + 
    geom_bar(position=position_dodge(), stat="identity") +
    geom_hline(yintercept = 1) + 
    geom_errorbar(aes(ymin=mean-se, ymax=mean+se), width=.2, position=position_dodge(.9)) +
    xlab("Recommendation Size") + ylab(yLabel) + ggtitle(title) +
    expand_limits(y=0) +
    theme_bw() +
    scale_x_discrete(labels=formatterPercent()) +
    #scale_fill_hue() + 
    theme(legend.justification=c(1,0), 
          legend.position="bottom",
          legend.title=element_text(size=12), 
          legend.text=element_text(size=8), 
          legend.key.size = unit(1,"line"),
          plot.title = element_text(size = 14, face = "bold"), 
          axis.title.x = element_text(size=10),
          axis.title.y = element_text(size=10)) 
}

getBarPlotCampaigns <- function(data, yLabel, title){
  ggplot(data, aes(x=CampaignId, y=mean, fill=CampaignId)) + 
    geom_bar(position=position_dodge(), stat="identity") +
    geom_hline(yintercept = 1) + 
    geom_errorbar(aes(ymin=mean-se, ymax=mean+se), width=.2, position=position_dodge(.9)) +
    xlab("CampaignId") + ylab(yLabel) + ggtitle(title) +
    expand_limits(y=0) +
    theme_bw() +
    scale_x_discrete() +
    scale_fill_hue() + 
    theme(legend.justification=c(1,0), 
          legend.position="bottom",
          legend.title=element_text(size=12), 
          legend.text=element_text(size=8), 
          legend.key.size = unit(1,"line"),
          plot.title = element_text(size = 14, face = "bold"), 
          axis.title.x = element_text(size=10),
          axis.title.y = element_text(size=10)) 
}

getBarPlotSimulation <- function(data, yLabel, title){
  ggplot(data, aes(x=SimulationId, y=mean, fill=SimulationId)) + 
    geom_bar(position=position_dodge(), stat="identity") +
    geom_hline(yintercept = 1) + 
    geom_errorbar(aes(ymin=mean-se, ymax=mean+se), width=.2, position=position_dodge(.9)) +
    xlab("SimulationId") + ylab(yLabel) + ggtitle(title) +
    expand_limits(y=0) +
    theme_bw() +
    scale_x_discrete() +
    scale_fill_hue() + 
    theme(legend.justification=c(1,0), 
          legend.position="bottom",
          legend.title=element_text(size=12), 
          legend.text=element_text(size=8), 
          legend.key.size = unit(1,"line"),
          plot.title = element_text(size = 14, face = "bold"), 
          axis.title.x = element_text(size=10),
          axis.title.y = element_text(size=10)) 
}
