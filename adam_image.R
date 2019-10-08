# Adam estimation image for manuscript titled
# "SEM using Computation Graphs"
# Last edited: 08-10-2019 by Erik-Jan van Kesteren

# load packages
library(tidyverse)
library(firatheme)

# the objective and its gradient
func <- function(x) x[1]^2 + 5*x[2]^2
grad <- function(x) c(2*x[1], 10*x[2])


# Gradient descent
alpha     <- 0.01
iters     <- 1000
gdesc     <- matrix(0, iters + 1, 2)
m         <- matrix(0, iters + 1, 2)
gdesc[1,] <- c(-0.9, -0.9)
colnames(gdesc) <- c("x", "y")

for (t in 2:(iters + 1)) {
  gt        <- grad(gdesc[t - 1,])
  gdesc[t,] <- gdesc[t - 1, ] - alpha * gt
}


# Gradient descent with momentum
decay     <- 0.9
theta     <- matrix(0, iters + 1, 2)
m         <- matrix(0, iters + 1, 2)
theta[1,] <- c(-0.9, -0.9)
colnames(theta) <- c("x", "y")

for (t in 2:(iters + 1)) {
  gt        <- grad(theta[t - 1,])
  m[t,]     <- decay * m[t - 1,] + (1 - decay) * gt
  theta[t,] <- theta[t - 1, ] - alpha * m[t,]
}


# Adam
decay2   <- 0.999
eps      <- 1e-8
adam     <- matrix(0, iters + 1, 2)
m        <- matrix(0, iters + 1, 2)
v        <- matrix(0, iters + 1, 2)
adam[1,] <- c(-0.9, -0.9)
colnames(adam) <- c("x", "y")

for (t in 2:(iters + 1)) {
  gt       <- grad(adam[t - 1,])
  m[t,]    <- decay * m[t - 1,] + (1 - decay) * gt
  v[t,]    <- decay2 * v[t - 1,] + (1 - decay2) * gt^2
  adam[t,] <- adam[t - 1, ] - alpha * m[t,] / (sqrt(v[t,]) + eps)
}


# Create the surface
surface   <- expand.grid(x = seq(-1, 1, len = 100), y = seq(-1, 1, len = 100))
surface$z <- apply(xyz, 1, objective)


# Create the path data frame
paths <- data.frame(
  rbind(gdesc, theta, adam), 
  type = rep(c("Gradient descent", "GD with momentum", "Adam"), each = 1001),
  iteration = rep(1:1001, 3)
)


# Plot
plt <- 
  ggplot(paths %>% filter(iteration < 400), aes(x = x, y = y, colour = type)) +
  geom_contour(data = surface, mapping = aes(z = z), col = "black", alpha = 0.2, 
               binwidth = 0.1) +
  geom_point() +
  coord_fixed() +
  xlim(-1, .5) +
  ylim(-1, .5) +
  labs(x = expression(theta[1]), y = expression(theta[2]), colour = "") +
  theme_fira() +
  scale_colour_fira() +
  theme(panel.grid.major = element_blank(), legend.position = "top")

firaSave(filename = "./img/adam_image.pdf", width = 6, height = 6, plot = plt)
firaSave(filename = "./img/tiff/adam_image.tiff", device = "tiff", width = 6, 
         height = 6, plot = plt, dpi = 300)

# Animation
library(gganimate)
anim <- 
  ggplot(paths %>% filter(iteration < 400), aes(x = x, y = y, colour = type)) +
  geom_contour(data = surface, mapping = aes(z = z), col = "black", alpha = 0.2, 
               binwidth = 0.1) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  coord_fixed() +
  xlim(-1, .5) +
  ylim(-1, .5) +
  labs(x = expression(theta[1]), y = expression(theta[2]), colour = "") +
  theme_fira() +
  scale_colour_fira() +
  theme(panel.grid.major = element_blank(), legend.position = "top") +
  transition_reveal(iteration)

anim_save("./img/optim.gif", anim, duration = 5.9, width = 500*300/96,
          height = 500*300/96, res = 300)




