path = "~/Documents/ETH/Thesis/1plusX/Data/RawData/Campaigns/809153/Processed/sorted_time_impressions.csv"
data <- read.csv(file=path, header=TRUE, sep=",")


#colnames(data)[colnames(data)=="V1"] <- "timestamp"
#hist(data$Timestamp, 100)

data$Timestamp = data$Timestamp / 1000
data$Timestamp = as.POSIXct(data$Timestamp, origin="1970-01-01")

hist(data$Timestamp, breaks = "hours", 
     col="blue", main = "Histogramm of Campaign",  
     xlab = "Timestamp", ylab = "Frequency", freq=TRUE)

clicked_users = unique(data[data$Click == TRUE, "UserHash"])
clicked_users_behavior = data[data$UserHash %in% clicked_users, ]

# 24438 - users that clicked
length(clicked_users)
# 285379 - total impressions for users that clicked
nrow(clicked_users_behavior)
# 14392075 - total impressions
nrow(data)
colnames(data)
user_all_behavior = aggregate(data$Click, by=list(data$UserHash), FUN=length)[2]
user_click_behavior = aggregate(clicked_users_behavior$Click, by=list(clicked_users_behavior$UserHash), FUN=length)[2]
user_click_behavior_click = aggregate(clicked_users_behavior$Click, by=list(clicked_users_behavior$UserHash), FUN=sum)[2]
user_click_behavior_mean = aggregate(clicked_users_behavior$Click, by=list(clicked_users_behavior$UserHash), FUN=mean)[2]

# users that clicked more than one time
user_click_behavior = user_click_behavior[user_click_behavior < 50 ]

hist(user_click_behavior, breaks=100,  
     col="blue", main = "Histogram of Impressions for users that clicked",  
     xlab = "Impressions", ylab = "Frequency", freq=TRUE)

hist(user_click_behavior_click$x,  
     col="blue", main = "Histogram of Impressions for users that clicked",  
     xlab = "Impressions", ylab = "Frequency", freq=TRUE)

hist(user_click_behavior_mean$x,  
     col="blue", main = "Histogram of Clicks % for users that clicked",  
     xlab = "Click %", ylab = "Frequency", freq=TRUE)

user_all_behavior= user_all_behavior[user_all_behavior < 50]
hist(user_all_behavior,  
     col="blue", main = "Histogramm of Impressions for users that clicked",  
     xlab = "Impressions", ylab = "Frequency", freq=FALSE)



stats_path = "~/Documents/ETH/Thesis/1plusX/Data/RawData/Campaigns/809153/Processed/user_statistics.csv"
users <- read.csv(file=stats_path, header=TRUE, sep=",")
sum(users$ClickCount)

click_users = users[users$ClickCount > 0, ]
click_users$CTR = click_users$ClickCount/click_users$TotalImpressions
click_users$TimeOfFirstClick = click_users$FirstClickIndex/click_users$TotalImpressions
# 24438 unique users that clicked 
nrow(click_users)
# 285379 impressions for users that clicked 
sum(click_users$TotalImpressions)
# 143800 impressions to discard if discarding impressions past click
sum(click_users$TotalImpressions - click_users$FirstClickIndex)

head(click_users)

hist(click_users$TimeOfFirstClick)

bound = click_users[click_users$TimeUntilClickSec/(60*60) < 5,] # first 5 hours
hist(bound$TimeUntilClickSec/60,breaks=100)

hist(click_users$TimeUntilClickSec/(60*60),breaks=100)
hist(click_users$ActiveTime/(60*60),breaks=100)

hist(users$ActiveTime/(60*60),breaks=100)

hist(users[users$TotalImpressions < 100,]$TotalImpressions,breaks=100)
hist(click_users[click_users$TotalImpressions < 100,]$TotalImpressions,breaks=100)
hist(click_users[click_users$TotalImpressions < 100,]$FirstClickIndex,breaks=100)

click_users$TimeUntilClickSec
a= hist(click_users$TimeUntilClickSec, freq=FALSE)
ggplot(data=click_users, aes(TimeUntilClickSec)) + 
  geom_histogram(bins = 10, color="blue", aes(y=..density..))

sum(a$density)
plot(density(click_users$TimeUntilClickSec))




a = c(750.5083162094999, 225.41701968964762, 154.7279936113521, 52.02692293460939, 41.74775406001103, 34.09945057598528, 31.490912575223003, 26.968169200878453, 25.048410149649627, 22.468646598645602, 20.830715547904653, 20.47102531283156, 19.619477107547407, 19.056620797725298, 17.241179259633487, 16.26840174978627, 15.587650416857954, 14.426016781728082, 13.89609177179443, 13.21773569076487, 12.604112221643174, 11.765575158903124, 11.63730789684413, 10.881279072105443, 10.527680939158657, 10.348027053352471, 10.233587624387528, 9.901503798156917, 9.625453595823716, 9.42228609176346, 9.300751037667556, 9.03873888981161, 8.866528060018236, 8.679336521444451, 8.586138084687093, 8.332236148736028, 8.035701492153732, 7.952083611427673, 7.695350161332431, 7.662793679322451, 7.37823491511851, 7.2551310617180444, 7.21679141329931, 6.8933036708654125, 6.788402524413966, 6.707898213028879, 6.543466610558803, 6.433973104301604, 6.295827914554089, 6.240287694272784, 6.196216091163959, 6.093848163139129, 5.964803149207031, 5.892415297351376, 5.836366439857908, 5.774140549179536, 5.7386525325017566, 5.600411883352324, 5.502213763982819, 5.441024599552905, 5.4188145164822075, 5.303190353662831, 5.285257103229788, 5.239977567611496, 5.149451863074189, 5.084926281746957, 5.024549870734878, 4.924098115255276, 4.839632026437136, 4.794570954555513, 4.78400981140294, 4.7128281010884425, 4.654883068567838, 4.542442076735115, 4.44102712556304, 4.368924649737658, 4.360074283457223, 4.303908111963755, 4.241910596974663, 4.207857373133765, 4.1359433971087425, 4.091885638030345, 4.019769554535566, 3.9969832438373474, 3.9225031620228927, 3.860942457521402, 3.827438925243144, 3.7731405983657473, 3.741735613520461, 3.6421341535910137, 3.611373511743317, 3.518301322476674, 3.4348302776840507, 3.3473131627872865, 3.308902385413677, 3.178520100357394, 3.0999893466012245, 3.0066191019774573, 2.8743042165461756, 2.844211787075548)