##                                      Convolutional Neural Networks                                        ##
##                                           Classification                                ##
##                                                                                                           ##

library(keras)
library(tidyverse)

train_dir <- file.path("C:/Users/Yashar/Desktop/Data Science Bootcamp/R programming/Week 11",'train')
validation_dir <- file.path("C:/Users/Yashar/Desktop/Data Science Bootcamp/R programming/Week 11",'validation')
test_dir <- file.path("C:/Users/Yashar/Desktop/Data Science Bootcamp/R programming/Week 11", "test")

## Normalize images by dividing all values to 255

train_generated_data <- keras::image_data_generator(rescale = 1/255)
validation_generated_data <- keras::image_data_generator(rescale = 1/255)
test_generated_data <- keras::image_data_generator(rescale = 1/255)

### flow images from directory

train_generator <- flow_images_from_directory(
  train_dir,
  train_generated_data,
  target_size = c(32, 32),
  batch_size = 10,
  class_mode = "categorical" 
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_generated_data,
  target_size = c(32, 32),
  batch_size = 10,
  class_mode = "categorical"
  
)

### defining model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 8, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(32, 32, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 16, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c('acc')
)

### fit model 

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 5,
  callbacks = callback_tensorboard("logs/run_a"),
  validation_data = validation_generator,
  validation_steps = 50)

### call tensorboard

tensorboard("logs/run_a")


##### use model to evaluate test images

test_generator <- flow_images_from_directory(
  test_dir,
  test_generated_data,
  target_size = c(32, 32),
  batch_size = 5,
  class_mode = "categorical"
)


model %>% evaluate_generator(test_generator, steps = 50)

