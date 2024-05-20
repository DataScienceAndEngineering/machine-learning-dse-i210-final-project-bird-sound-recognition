# Revolutionizing Bird Sound Classification: Leveraging Machine Learning and Deep Learning Techniques

## Abstract

Bird species identification is vital for ecological monitoring and conservation efforts. Traditional methods, which rely on expert knowledge to identify bird species based on their vocalizations, are labor-intensive and time-consuming. In this report, we investigate the application of machine learning (ML) and deep learning (DL) models to automate bird sound classification. We leveraged audio features such as Mel Frequency Cepstral Coefficients (MFCC), Mel spectrogram, Log Mel Spectrograms, and waveform. Visual inspections of spectrograms were conducted to confirm the similarity of waveforms among species.

Our study compared the performance of various ML models, including Random Forest (RF), k-Nearest Neighbors (KNN), and XGBoost, alongside DL models such as Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN). The results indicate that ML models achieved significant test accuracies with MFCC features: RF at 90.14%, KNN at 85%, and XGBoost at 89.1%. Notably, the CNN model achieved a test accuracy of 95.51%, and the combination of CNN and RNN models achieved an even higher accuracy of 96.68%.

These findings demonstrate the potential of integrating ML and DL techniques to enhance the accuracy and efficiency of bird sound classification systems, offering a promising solution to support ecological monitoring and conservation initiatives.

## Introduction

Birds serve as essential indicators of environmental health and biodiversity, making their accurate identification vital for ecological research and conservation efforts. Traditional methods of identifying bird species through their vocalizations are time-consuming and demand specialized expertise, which limits their practicality for large-scale studies. With the rapid advancements in machine learning (ML) and deep learning (DL) technologies, there is significant potential to automate and enhance the process of bird sound classification.

Our research investigates the application of ML and DL models to automate bird sound classification. We posed the following research questions: How effective are ML and DL models in classifying bird sounds? Which models provide the highest accuracy? To address these questions, we compared the performance of various ML models, including Random Forest (RF), k-Nearest Neighbors (KNN), and XGBoost, as well as DL models like Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN).

Our key results demonstrate that the RF model achieved a test accuracy of 90.14%, KNN reached 85%, and XGBoost attained 89.1% using Mel Frequency Cepstral Coefficients (MFCC) features. Notably, the CNN model achieved a test accuracy of 95.51%, and the combined CNN and RNN models achieved 96.68%. These results underscore the potential of DL models to significantly enhance the accuracy of bird sound classification systems.

Our research plan involved the following steps: data collection and preprocessing, feature extraction, model training and evaluation, and result analysis. Exploratory data analysis was performed, and audio cleaning was done by applying the Fourier transformation to spectrograms, converting them to the time domain. This process yielded real and imaginary parts, which we combined to calculate the magnitude. We then computed the mean and standard deviation of these magnitudes and stored them in an array for further analysis. We utilized audio features like MFCC, Log Mel Spectrograms, and Waveform to capture various aspects of bird sounds.

To further understand the neural network's decision-making process, we used saliency maps. Saliency maps help us visualize where the neural network is focusing in particular while making a prediction. By examining these maps, we gained insights into which parts of the audio spectrogram were most influential in the network's classification decisions.

By systematically comparing the performance of different models, we aimed to develop a robust system capable of accurately identifying bird species from their vocalizations. This report provides a comprehensive analysis of our methodology, findings, and the implications of using ML and DL techniques for bird sound classification, offering valuable insights for ecological monitoring and conservation efforts.


## Background

Previous research has shown the effectiveness of machine learning models in various classification tasks, including speech and audio processing. Bird sound classification has been explored using different approaches, such as support vector machines (SVM) and random forest(RF), with varying degrees of success. Recent studies have highlighted the potential of convolutional neural networks (CNNs) in processing audio spectrograms for classification tasks, achieving high accuracy rates.

In this project, we build on these findings by implementing and comparing the performance of RF, KNN, XGBoost,CNN and RNN models. We utilize features like Mel Frequency Cepstral Coefficients (MFCC), Mel spectrogram, Log Mel Spectrograms and waveform, which have proven effective in audio processing tasks. By integrating these features with advanced classification algorithms, we aim to develop a robust system for classifying bird sounds. Identifying bird species through their sounds has been a subject of extensive research, with various methodologies and datasets being explored. 

Balemarthy et al. (2018) explored species recognition through vocalizations using a combination of advanced signal processing techniques and machine learning algorithms. They utilized Mel Frequency Cepstral Coefficients (MFCC) and Hidden Markov Models (HMM) to analyze continuous acoustic signals. Their approach achieved an overall accuracy of 66.7% in detecting species' sounds, highlighting the effectiveness of their methodology in noisy and uncontrolled environments.

Jadhav et al. (2020) conducted a study using bird audio data from the Macaulay Library, eBird, and xeno-canto. They evaluated various algorithms, including K-Nearest Neighbors (KNN), Random Forest, Multilayer Perceptron (MLP), and Bayes, with the Support Vector Machine (SVM) using a radial basis function kernel achieving an accuracy of 96.7%. Their methodology involved segmenting audio files into 3-second clips and using Principal Component Analysis (PCA) for dimensionality reduction.

Mehyadin et al. (2021) utilized the "Female Feature MFCC" dataset from Kaggle, initially designed for predicting emotions, to classify bird sounds. Their study applied algorithms like Naive Bayes, J4.8 (a decision tree algorithm), and Multilayer Perceptron (MLP), achieving the highest accuracy with J4.8 at 78.40%. They emphasized the importance of preprocessing, including noise reduction and filtering, to enhance the quality of sound recordings and the effectiveness of MFCC in capturing essential features for classification.

Xiao et al. (2022) proposed an automatic recognition model of bird sounds. The proposed model, AMResNet, is based on the combination of an attentional mechanism and residual networks, which can automatically extract and select high-dimensional features to achieve high classification accuracy. The attentional mechanism improves recognition efficiency by assigning appropriate weights to channels and space, while the residual network alleviates the problem of gradient disappearance and increases information flow through skip connections. More importantly, an efficient combined feature representation was adopted, providing a more comprehensive depiction of bird sounds.

The AMResNet model was trained and tested using 10-fold cross-validation on 12,651 bird sound samples from real environments. The results demonstrated that AMResNet significantly outperformed eight other models, including two traditional models, a forward neural network, four main deep learning models, and the latest vision transformer model. The proposed model achieved a classification accuracy of 92.6%, which was 3.1% higher than the best of the other eight models. It also achieved the best results in precision (97.6%), recall (97.3%), and F1-score (97.1%) across the 19 species.

In our project, we also explored the combination of advanced features for bird sound classification. By combining log mel spectrograms and MFCCs, we achieved higher accuracy with models like Random Forest and XGBoost compared to using log mel spectrograms individually.

Yang et al. (2023) introduced SSL-Net, a dual-branch structure for bird sound classification, utilizing audio-pretrained models and traditional acoustic feature extraction techniques. They achieved the accuracy of 85.70% using a sampling fusion strategy that combined spectral features and pretrained models. Their study demonstrated the potential of integrating spectral and learned features to enhance classification robustness.

These studies highlight the effectiveness of various machine learning and deep learning models in bird sound classification. By leveraging advanced audio processing techniques and robust classification algorithms, it is possible to develop highly accurate and efficient systems for automatic bird sound identification.


## Citations

Balemarthy, S., Sajjanhar, A., & Zheng, J. X. (2018). Our practice of using machine learning to recognize species by voice. https://doi.org/10.48550/ARXIV.1810.09078

Jadhav, Y., Patil, V., & Parasar, D. (2020). Machine learning approach to classify birds on the basis of their sound. 2020 International Conference on Inventive Computation Technologies (ICICT), 69–73. https://doi.org/10.1109/ICICT48043.2020.9112506

Mehyadin, A. E., Abdulazeez, A. M., Hasan, D. A., & Saeed, J. N. (2021). Birds sound classification based on machine learning algorithms. Asian Journal of Research in Computer Science, 1–11. https://doi.org/10.9734/ajrcos/2021/v9i430227

Xiao, H., Liu, D., Chen, K., & Zhu, M. (2022). AMResNet: An automatic recognition model of bird sounds in real environment. Applied Acoustics, 201, 109121. https://doi.org/10.1016/j.apacoust.2022.109121

Yang, Y., Zhou, K., Trigoni, N., & Markham, A. (2023). Ssl-net: A synergistic spectral and learning-based network for efficient bird sound classification. https://doi.org/10.48550/ARXIV.2309.08072

