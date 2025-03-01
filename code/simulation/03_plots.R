require("ggplot2")
require("tidyverse")

# ROOT = "C:/Users/jvavra/OneDrive - WU Wien/Documents/TVTBIP/"
ROOT = "C:/Users/Vavra/PycharmProjects/TVTBIP/"
DAT = paste0(ROOT, "data/")
SIM = paste0(DAT, "simulation/")
FIG = paste0(SIM, "plot/")
data_names = c('simulation-zero', 'simulation-party',
               'simulation-diverge', 'simulation-estimate')
if(!dir.exists(FIG)){
  dir.create(FIG)
}

max_session = 114

DF3 <- part <- list()

for(data_name in data_names){

  DF <- read.csv(paste0(SIM, data_name, "_ideal_points_all_sessions_rescaledIQR.csv"),
                 header = TRUE, sep = ",")

  #colnames(DF)[1] <- "Party"
  ## index Republican and Democrat
  DF$Party <- rep("Democrats", nrow(DF))
  ind <- grep("\\(R\\)", DF$speaker)
  DF$Party[ind] <- "Republicans"
  ind <- grep("\\(I\\)", DF$speaker)
  DF$Party[ind] <- "Independent"

  colnames(DF) <- gsub("ideal_point_", "ip", colnames(DF))
  DF2  <- DF |>
    pivot_longer(cols = ip97:paste0("ip",max_session),
                 names_to = "Session",
                 values_to = "IdealPoint",
                 names_prefix = "ip",
                 values_drop_na = TRUE)

  # convert session to numeric
  DF2$Session <- factor(DF2$Session, levels = 97:max_session, ordered = TRUE)

  session_to_year <- function(session) {
    a <- 1981 + 2*(as.numeric(session) - 1)
    y <- paste(a, a+1, sep="-")
    return(y)
  }

  session_to_year2 <- function(session) {
    a <- 1787 + 2*as.numeric(session)
    y <- paste(a, a+1, sep="-")
    return(y)
  }

  year_levels <- session_to_year2(levels(DF2$Session))
  year_levels <- paste0(year_levels, "\n(", levels(DF2$Session), ")")
  year_levels <- gsub("^(19|20)", "", year_levels)
  year_levels <- gsub("-(19|20)", "-", year_levels)

  #DF2$Year <- session_to_year(DF2$Session)
  #DF2$Year <- paste0(DF2$Year, "\n(", DF2$Session, ")")
  #DF2$Year <- gsub("^(19|20)", "", DF2$Year)
  #DF2$Year <- gsub("-(19|20)", "-", DF2$Year)
  DF2$Year <- factor(DF2$Session, levels=levels(DF2$Session), labels=year_levels)

  # Ignore Independent senators
  DF3[[data_name]] <- subset(DF2, Party != "Independent")

  ggplot(DF3[[data_name]], aes(x = Year, y = IdealPoint, fill = Party)) +
    geom_boxplot(outliers = FALSE) +
    scale_fill_manual(values = c("blue", "red")) +
    ylab("Ideal Point") +
    xlab("Years (Session)") +
    ylim(-1, 1) +
    theme_bw() +
    theme(legend.position.inside = c(0.005, 0.99),
          legend.justification = c("left", "top"),
          legend.title = element_blank())

  ggsave(paste0(FIG, data_name, "_boxplots_R_vs_D_rescaledIQR.pdf"),
         width = 20, height = 14, units = "cm")

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

    for (i in 97:max_session) {
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
  Partisan$Session <- factor(97:max_session, levels = 97:max_session, ordered = TRUE)
  Partisan$Year <- session_to_year(Partisan$Session)
  Partisan$Year <- paste0(Partisan$Year, "\n (", Partisan$Session, ")")
  Partisan$Year <- gsub("^(19|20)", "", Partisan$Year)
  Partisan$Year <- gsub("-(19|20)", "-", Partisan$Year)
  Partisan$Year <- factor(Partisan$Year, Partisan$Year)

  part[[data_name]] <- Partisan

  ## plot with grey CI interval
  ggplot(data = Partisan, aes(x = Year, y = Partisanship, group = 1)) +
    geom_ribbon(aes(ymin = Lower, ymax = Upper), fill = "lightgrey") +
    geom_line() +
    geom_point() +
    ylab("Average Partisanship") +
    xlab("Years (Session)") +
    scale_y_continuous(limits = c(-0.15, 1.35)) +
    theme_bw()

  ggsave(paste0(FIG, data_name, "_partisanship_rescaledIQR.pdf"),
         width = 20, height = 14, units = "cm")
}

part

scenarios <- c("Zero ideals", "Party-fix ideals",
               "Diverging ideals", "Estimated ideals")
names(scenarios) <- data_names
final_part <- data.frame()
for(data_name in data_names){
  part[[data_name]]$Scenario <- scenarios[data_name]
  # part[[data_name]]$true_difference <-
  final_part <- rbind(final_part, part[[data_name]])
}
final_part

ggplot(data = final_part, aes(x = Year, y = Partisanship, group = Scenario, color = Scenario)) +
  #geom_ribbon(aes(ymin = Lower, ymax = Upper), fill = "grey95") +
  geom_line() +
  geom_point() +
  ylab("Average Partisanship") +
  xlab("Years (Session)") +
  scale_y_continuous(limits = c(-0.15, 1.35)) +
  theme_bw()

ggsave(paste0(FIG, "partisanship_simulations_together_rescaledIQR.pdf"),
       width = 25, height = 14, units = "cm")
