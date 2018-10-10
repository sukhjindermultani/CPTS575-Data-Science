#CptS 475/575: Data Science, Fall 2018
##Assignment 2: R basics and Exploratory Data Analysis
###Release Date: September 10, 2018 Due Date: September 17, 2018 (11:59 pm)


college <- read.csv("https://scads.eecs.wsu.edu/wp-content/uploads/2017/09/College.csv")
# View(college)
rownames(college) <- college[, 1]
fix(college)

college <- college[, -1]
  fix(college)
summary(college)


pairs(college[, 1:10])

length(college$Private)

plot(college$Private, college$Outstate,
     xlab = "Private University", ylab = "Tuition in $")


Top <- rep("No", nrow(college))
Top[college$Top25perc > 50] <- "Yes"
Top <- as.factor(Top)
college <- data.frame(college, Top)

summary(college)

plot(college$Outstate, college$Top,
     xlab = "Outstate", ylab = "Top",title("Top Universities"))


par(mfrow=c(2,2))
hist(college$Apps, xlab = "Applications Received", main = "")
hist(college$perc.alumni, col=2, xlab = "% of alumni who donate", main = "")
hist(college$S.F.Ratio, col=3, breaks=10, xlab = "Student/faculty ratio", main = "")
hist(college$Expend, breaks=100, xlab = "Instructional expenditure per student", main = "")



# Some interesting observations:

# what is the university with the most students in the top 10% of class
row.names(college)[which.max(college$Top25perc)]  

acceptance_rate <- college$Accept / college$Apps

# what university has the smallest acceptance rate
row.names(college)[which.min(acceptance_rate)]  

# what university has the most liberal acceptance rate
row.names(college)[which.max(acceptance_rate)]

# High tuition correlates to high graduation rate
plot(college$Outstate, college$Grad.Rate) 

# Colleges with low acceptance rate tend to have low S:F ratio.
plot(college$Accept / college$Apps, college$S.F.Ratio) 

# Colleges with the most students from top 10% perc don't necessarily have
# the highest graduation rate. Also, rate > 100 is erroneous!
plot(college$Top25perc, college$Grad.Rate)




#########################################################################################

Auto <- read.csv("http://www-bcf.usc.edu/~gareth/ISL/Auto.csv", 
                 header = TRUE, na.strings = "?")

Auto <- na.omit(Auto)
dim(Auto)
summary(Auto)

Auto$originf <- factor(Auto$origin, labels = c("usa", "europe", "japan"))
with(Auto, table(originf, origin))


#Pulling together qualitative predictors
qualitative_columns <- which(names(Auto) %in% c("name", "origin", "originf"))
qualitative_columns

# Apply the range function to the columns of Auto data
# that are not qualitative
sapply(Auto[, -qualitative_columns], range)


sapply(Auto[, -qualitative_columns], mean)
sapply(Auto[, -qualitative_columns], sd)


sapply(Auto[-seq(10, 85), -qualitative_columns], mean)
sapply(Auto[-seq(10, 85), -qualitative_columns], sd)



# Part (e):
pairs(Auto)
#Quantitative Columns
pairs(Auto[, -qualitative_columns])
#Qualitative Columns
pairs(Auto[, qualitative_columns])

# Heavier weight correlates with lower mpg.
with(Auto, plot(mpg, weight))


# More cylinders, less mpg.
with(Auto, plot(mpg, cylinders))

# Cars become more efficient over time.
with(Auto, plot(mpg, year))

# Lets plot some mpg vs. some of our qualitative features: 
# sample just 20 observations
Auto.sample <- Auto[sample(1:nrow(Auto), 20), ]

# order them
Auto.sample <- Auto.sample[order(Auto.sample$mpg), ]

# plot them using a "dotchart"
with(Auto.sample, dotchart(mpg, name, xlab = "mpg"))
with(Auto, plot(originf, mpg), ylab = "mpg")


pairs(Auto)