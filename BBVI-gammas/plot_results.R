library(ggplot2)
library(reshape2)

# load data and extract the known groudn truth
dat <- read.csv('log.tsv', sep='\t')

truth <- dat[dat$iteration==-1,]
dat <- dat[dat$iteration!=-1,]

truth[truth$mean==0,]$mean <- 1e-10
truth[truth$sparsity==0,]$sparsity <- 1e-10

# plot the data with known means
p <- ggplot(dat, aes(x=iteration, y=mean)) + facet_wrap(~ component, scales="free") +theme_bw()
p <- p  + geom_line() +geom_hline(data=truth, aes(yintercept=mean),color="red")
p <- p + geom_point()

ggsave("results.pdf", p, height=7, width=14)
