# Image Segmentation with U-Net and Hyperparameter Tuning

## Description

This project focuses on performing image segmentation using a U-Net architecture. The goal is to classify each pixel of an image into one of six predefined classes. The project involves loading a dataset of satellite images and their corresponding masks, preprocessing the data through resizing, augmentation, shuffling, one-hot encoding, and normalization. Hyperparameter tuning and nested cross-validation are employed to find the optimal model configuration.

## Data

The dataset is loaded from the pickle file hosted on the GitHub repository. It consists of satellite images and their corresponding segmentation masks.

*   **Initial Shape:** The images have an initial shape of (1305, 256, 256, 3) and the masks have a shape of (1305, 256, 256, 1).
*   **Resized Shape:** The images and masks are resized to (128, 128). After resizing and augmentation, the dataset contains 8 times the original number of samples. The final shapes are (10440, 128, 128, 3) for images and (10440, 128, 128, 6) for one-hot encoded masks.

## Methodology

The following steps were taken to preprocess and prepare the data:

1.  **Resizing:** Images and masks are resized from 256x256 to 128x128 using OpenCV.
2.  **Data Augmentation:** The dataset is augmented using `imgaug` by applying rotations (90°, 180°, 270°), horizontal and vertical flips, and zooming (by 20%). This increases the dataset size significantly.
3.  **Shuffling:** The augmented dataset is shuffled to prevent the model from learning based on the order of samples.
4.  **One-Hot Encoding:** The segmentation masks, which contain integer labels from 0 to 5, are converted into a one-hot encoded format using `to_categorical`. The number of classes is 6.
5.  **Normalization:** The image pixel values are scaled to the range [0, 1] by dividing by 255.
6.  **Train-Test Split:** The dataset is split into training and testing sets with a test size of 20%.

## Model

The project utilizes a U-Net architecture for image segmentation.

*   **Basic U-Net:** A simple U-Net model was initially built and tested to ensure the architecture and training pipeline were functioning correctly. This model had fewer layers and filters.

*   **Hyperparameter Tuned U-Net:** A more complex U-Net architecture is used for hyperparameter tuning. This model includes:
    *   Encoder and Decoder paths with multiple convolutional and pooling/upsampling layers.
    *   Skip connections between corresponding encoder and decoder layers.
    *   `relu` activation function for hidden layers.
    *   `softmax` activation function for the output layer to predict class probabilities for each pixel.
    *   `Dropout` layers and `L2 regularization` are incorporated to mitigate overfitting.
*   **Loss Function:** `categorical_crossentropy` is used as the loss function, suitable for multiclass classification with one-hot encoded targets.
*   **Optimizer:** The `Adam` optimizer is used for training.

  <img width="800" height="497" alt="Unknown-3" src="https://github.com/user-attachments/assets/2a083a91-e5f2-41ae-9fda-db50c008b7f7" />

## Results

Hyperparameter tuning was performed using `RandomSearch` from `keras-tuner` in conjunction with nested cross-validation.

*   **Hyperparameter Tuning:** The optimal hyperparameters for the learning rate, batch size, dropout rate, number of filters in convolutional layers, and L2 regularization weight were determined.
*   **Nested Cross-Validation:** Nested cross-validation with 2 outer and 2 inner splits was used to evaluate the model's performance and provide a more robust estimate of its generalization ability.
*   **Final Accuracy:** The best model found through hyperparameter tuning was trained on the entire training dataset. The final test accuracy achieved is approximately 0.8.
*   **Learning Curve:** Plots showing the training and validation loss and accuracy over epochs are generated to visualize the training progress and assess for overfitting.
*   **Visualization of Masks:** Sample images, their true masks, and the masks predicted by the best model are displayed to qualitatively assess the model's performance.

<img width="831" height="418" alt="Unknown" src="https://github.com/user-attachments/assets/d74bf6e2-39cc-40c3-ac65-7f8397c2cfd2" />

## Conclusion

This project successfully implemented a U-Net model for image segmentation on the provided dataset. Through comprehensive data preprocessing, including augmentation and one-hot encoding, and the application of hyperparameter tuning with nested cross-validation, an optimal model configuration was identified. The model achieved a test accuracy of around 0.8, and the visualizations of predicted masks demonstrate its effectiveness in segmenting the images into different classes. The results indicate that the developed model is suitable for practical image segmentation tasks.
