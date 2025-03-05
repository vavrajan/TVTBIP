require("ggplot2")
require("tidyverse")

ROOT <- getwd() # set your own working directory
FIG <- paste0(ROOT, "/TVTBIP/data/hein-daily-plot/")

DF <- read.csv(paste0(FIG, "ideal_points_all_sessions.csv"), header = TRUE, sep = ",")

#colnames(DF)[1] <- "Party"
## index Republican and Democrat
DF$Party <- rep("Democrats", nrow(DF))
ind <- grep("\\(R\\)", DF$senator)
DF$Party[ind] <- "Republicans"

colnames(DF) <- gsub("ideal_point_", "ip", colnames(DF))
DF2  <- DF |>
    pivot_longer(cols = ip97:ip114,
                 names_to = "Session",
                 values_to = "IdealPoint",
                 names_prefix = "ip",
                 values_drop_na = TRUE)

# convert session to numeric
DF2$Session <- factor(DF2$Session, levels = 97:114, ordered = TRUE)

session_to_year <- function(session) {
  a <- 1981 + 2*(as.numeric(session) - 1)
  y <- paste(a, a+1, sep="-")
  return(y)
}

DF2$Year <- session_to_year(DF2$Session)
DF2$Year <- paste0(DF2$Year, "\n(", DF2$Session, ")")
DF2$Year <- gsub("^(19|20)", "", DF2$Year)
DF2$Year <- gsub("-(19|20)", "-", DF2$Year)
DF2$Year <- factor(DF2$Year, unique(DF2$Year))

ggplot(DF2, aes(x = Year, y = IdealPoint, fill = Party)) +
    geom_boxplot(outliers = FALSE) +
    scale_fill_manual(values = c("blue", "red")) +
    ylab("Ideal Point") +
    xlab("Years (Session)") +
    ylim(-1, 1) +
    theme_bw() +
    theme(legend.position = c(0.005, 0.99),
          legend.justification = c("left", "top"),
          legend.title = element_blank())

ggsave(paste0(FIG, "Rep_vs_Dem_in_time.pdf"), width = 20, height = 14, units = "cm")

# Switch Rep and Dem
DF3 <- DF2
DF3$IdealPoint <- (-1)*DF3$IdealPoint
ggplot(DF3, aes(x = Year, y = IdealPoint, fill = Party)) +
  geom_boxplot(outliers = FALSE) +
  scale_fill_manual(values = c("blue", "red")) +
  ylab("Ideal Point") +
  xlab("Years (Session)") +
  ylim(-1, 1) +
  theme_bw() +
  theme(legend.position = c(0.005, 0.99),
        legend.justification = c("left", "top"),
        legend.title = element_blank())

ggsave(paste0(FIG, "Dem_vs_Rep_in_time.pdf"), width = 20, height = 9, units = "cm")

########## estimate partisanship for each session

DF_R <- DF |>
    filter(Party == "Republicans")
DF_D <- DF |>
    filter(Party == "Democrats")

partisan.function <- function(X, Y) {
  n <- nrow(X)
  m <- nrow(Y)
  partisanship <- numeric(0)
  upper_b <- numeric(0)
  lower_b <- numeric(0)

  for (i in 97:114) {
    ip_R <- X[, paste0("ip",i)]
    ip_D <- Y[, paste0("ip",i)]
    partisanship[i-96] <- abs(mean(ip_R, na.rm = TRUE) - mean(ip_D,, na.rm = TRUE))

    n_D <- length(ip_D)-sum(is.na(ip_D))
    n_R <- length(ip_R)-sum(is.na(ip_R))
    lower_b[i-96] <- partisanship[i-96] - 1.96 * sqrt((var(ip_R, na.rm = TRUE)/n_R) + (var(ip_D, na.rm = TRUE)/n_D))
    upper_b[i-96] <- partisanship[i-96] + 1.96 * sqrt((var(ip_R, na.rm = TRUE)/n_R) + (var(ip_D, na.rm = TRUE)/n_D))
  }

  CI <- data.frame(Lower = lower_b,
                   Partisanship = partisanship,
                   Upper = upper_b)
  return(CI)
}

Partisan <- partisan.function(DF_R, DF_D)
Partisan$Session <- factor(97:114, levels = 97:114, ordered = TRUE)
Partisan$Year <- session_to_year(Partisan$Session)
Partisan$Year <- paste0(Partisan$Year, "\n (", Partisan$Session, ")")
Partisan$Year <- gsub("^(19|20)", "", Partisan$Year)
Partisan$Year <- gsub("-(19|20)", "-", Partisan$Year)
Partisan$Year <- factor(Partisan$Year, Partisan$Year)

## plot with grey CI interval
ggplot(data = Partisan, aes(x = Year, y = Partisanship, group = 1)) +
    geom_ribbon(aes(ymin = Lower, ymax = Upper), fill = "lightgrey") +
    geom_line() +
    geom_point() +
    ylab("Average Partisanship") +
    xlab("Years (Session)") +
    theme_bw()

ggsave(paste0(FIG, "partisanship_in_time.pdf"), width = 20, height = 14, units = "cm")


