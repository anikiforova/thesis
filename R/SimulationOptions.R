
library(scales)
data = data.frame(x=double(), type=integer())
for (i in c(1, 5, 10, 20, 100, 100000)) {
   x =  rchisq(10000, df=i, ncp = 0) 
#  x = x[x <= 1]
  x = 2*(x - min(x)) / (max(x) - min(x))
    data1 = data.frame(x = x, type=i)
  data = rbind(data, data1)
}

ggplot(data, aes(x=x, color=factor(type))) +
  geom_density(aes(y = ..density..))
  
data = data.frame(x=double(), type=integer())
for (df in c(1, 10, 100)) {
  x =  rchisq(100, df=df, ncp = 0) 
  for (alpha in c(1, 2, 4)) {
    y = alpha*(x - min(x)) / (max(x) - min(x))
    
    data1 = data.frame(x = y, type=paste(df, alpha))
    data = rbind(data, data1)  
  }
}

ggplot(data, aes(x=x, color=factor(type))) +
  geom_density(aes(y = ..density..)) + xlim(c(0, 1))


data = data.frame(x=double(), type=integer())
for (i in seq(2, 20, 4)) {
  x = rbeta(100, 1, i, ncp = 0) 
  #  x = x[x <= 1]
  x = (x - min(x)) / (max(x) - min(x))
  data1 = data.frame(x = x, type=i)
  data = rbind(data, data1)
}


ggplot(data, aes(x=x, color=factor(type))) +
  geom_density(aes(y = ..density..))



x = rchisq(10000, df=0, ncp = 0) ?rchisq