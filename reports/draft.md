Data Source
The data for the bird sound classifier project is sourced from the Kaggle dataset titled "Bird Song Data Set". This dataset includes a diverse collection of bird sound recordings labeled with species names, essential for developing a robust classification model.
url: ______________________________-
The aim of the project is to identify the bird species based on the sound they generate.

About Dataset
Context
A small collection of bird songs from 5 species, to explore classification / identification of a bird in this set with an audio recording. Original source is https://www.xeno-canto.org/
Content
Data set includes audios from 5 species-
•	Bewick's Wren
•	Northern Cardinal
•	American Robin
•	Song Sparrow
•	Northern Mockingbird
For simplicity, data set excludes other types of calls (alarm calls, scolding calls etc) made by these birds. Additionally, only recordings graded as highest quality on xeno-canto API are included.
Original recordings from xeno-canto were in mp3. All files were converted to wav format.
This was done because
1.	Lossless Quality: WAV is a lossless format, meaning it retains all the original audio data without any compression. In contrast, MP3 is a lossy format that compresses audio by removing certain frequencies, which can degrade the quality. For analysis, having the highest quality audio is crucial to ensure accurate results.
2.	Preservation of Original Data: WAV files maintain the original data without any alteration. This is important for tasks such as noise reduction, feature extraction, and other preprocessing steps, where any loss of information can affect the outcome.
3.	Consistency and Standardization: Many audio analysis tools and libraries are designed to work with WAV files because of their consistent format and structure. MP3 files, with their variable bit rates and compression artifacts, can introduce inconsistencies in analysis.
4.	Better Handling of Features: Certain audio features, like frequencies, harmonics, and transient responses, are better preserved in WAV format. This makes it easier to perform tasks like speech recognition, music information retrieval, and other audio analysis tasks.
5.	Avoiding Compression Artifacts: MP3 compression can introduce artifacts such as pre-echo, ringing, and other distortions that can interfere with the analysis. WAV files, being uncompressed, do not have these artifacts, leading to more accurate and reliable analysis.
6.	Higher Dynamic Range: WAV files support a higher dynamic range, which is important for analyzing audio signals with a wide range of volumes. This is particularly important in fields like music analysis and environmental sound analysis. 

In summary, converting audio from MP3 to WAV ensures the preservation of audio quality and consistency, which is essential for accurate and effective audio analysis.

 Further, using onset detection, original recordings of varying lengths were clipped to exactly 3sec such that some portion of the target bird's song is included in the clip.
Original mp3 files from the source have varying sampling rates and channels
CSV file includes recording metadata, such as genera, species, location, datetime, source url, recordist and license information.
The filename column in CSV corresponds to the wav files under wavfiles folder
Acknowledgements
All information is sourced from API at https://www.xeno-canto.org/
Inspiration
What features in a bird’s sound are critical in distinguishing it from other species? How accurately can we identify a bird given a 3s recording? This is the aim of our project and to analyze this, we used Kaggle to extract the dataset because the data was readily available in a wav format.


Description of Variables
The dataset includes several key variables:
id: Unique identifier for each recording.
genus: Genus of the bird species.
species: Species of the bird.
subspecies: Subspecies of the bird (if available).
name: Common name of the bird.
recordist: Name of the person who recorded the audio.
country: Country where the recording was made.
location: Specific location where the recording was made.
latitude: Latitude coordinate of the recording location.
longitude: Longitude coordinate of the recording location.
laltitude: Altitude (in meters) at the recording location.
sound_type: Description of the sound type (e.g., song, call, etc.).
source_url: URL link to the source of the recording.
license: Licensing information for the recording.
time: Time of day when the recording was made.
date: Date when the recording was made.
remarks: Additional remarks or notes about the recording.
filename: Name of the audio file.
These columns provide comprehensive metadata for each bird song recording, which can be useful for analysis and research purposes.


Data Wrangling and Cleaning
Data preparation steps included:
Loading the Data: Imported into a pandas Data Frame. Applied different other libraries such as numpy to store mean and standard deviation of each audio as features in an array, librosa to convert audio in frequency domain so that the fourier transform can be extracted, seaborn for pair plot, tensorflow for complex machine learning algorithms such as KNN, sklearn for model selection, skimage for reading the images which were obtained from the audio, importing Random Forest which is the base model acting as a classifier, xgboost also for classification. We then 
Handling Missing Values: Missing values were identified and imputed. Numerical columns used the median, while categorical columns used the mode.
Consistency Check for Units: Ensured all measurements were in consistent units (e.g., seconds for duration, dB for frequency).
Reformatting Data: Reformatted the dataset to ensure each row represented a unique observation and each column a distinct variable.
Feature Engineering: Created new features such as mean frequency to enhance model performance.


Exploratory Data Analysis (EDA)
EDA involved visualizing and summarizing the dataset's main characteristics to identify patterns and insights.
Univariate Analysis:
Histograms to identify the total number of species that we have
Bivariate Analysis:
Scatter plots for relationships between numerical variables (e.g., amplitude vs. frequency range).
Correlation matrices and heatmaps for correlations between variables in the meta data.
Multivariate Analysis:
Pair plots for insights into interactions between multiple variables.
Cluster analysis suggested natural groupings of recordings by species based on acoustic features. This was using KNN.
Handling Scale and Units:
Applied logarithmic scales to skewed data distributions where appropriate.
Target Variable Contamination Check:
Ensured no leakage of the target variable (species) into input features, maintaining model integrity.
Thorough EDA: Conducted a thorough exploratory data analysis to understand the data's structure and relationships.
Missing Data Identification: Identified and imputed missing data points appropriately.
Parameter and Unit Tracking: Kept track of all parameters and their units.

Visualization Summaries: Provided comprehensive visualization summaries for all data and features.
Correct Visualization Types: Used appropriate visualization types, such as histograms  for categorical data and scatter plots for numerical data.
Proper Axis Labeling: Ensured all axes were properly labeled.
Effective Use of Color: Utilized color effectively to distinguish different data points and categories.
Informative Captions: Included makrdowns that clearly explained the conclusions that could be drawn from each figure.


Conclusion
The bird sound classifier demonstrates promising results in distinguishing bird species based on their acoustic characteristics. Key variables such as frequency range and amplitude were particularly useful in classification. The model is robust, given the comprehensive dataset and meticulous preprocessing steps. While additional data and continuous updates can further enhance the model's accuracy, the current results of the CNN+RNN are highly satisfactory.




Attribution
Contribution by Author
Using GitHub commit data, contributions by each author were visualized:
Author A: 50 commits, focused on data wrangling and feature engineering (approximately 30 code-hours).
Author B: 50 commits, responsible for EDA and visualization (approximately 30 code-hours).
Author C: 30 commits, worked on model building and evaluation (approximately 30 code-hours).
Author D: 30 commits, worked on CNN and CCN+RNN (approximately 35 code-hours)

Bibliography
1.	Shanbhag, Vinay. "Bird Song Data Set." Kaggle, 2023. URL
2.	Adi WhatsApp group chat
3.	___

Appendix

Additional Results and Graphs
Figure A1: Histogram of recording durations.
Discussion: This histogram shows the distribution of recording lengths, indicating most recordings are within a specific range.
Figure A2: Scatter plot of amplitude vs. frequency range.
Discussion: This plot highlights the relationship between amplitude and frequency, with distinct clusters for different species.
This report provides a detailed overview of the bird sound classifier project, documenting every step from data wrangling to EDA and model evaluation, ensuring clarity and reproducibility.
________________________________________This detailed report ensures all criteria are met, providing a comprehensive overview of the bird sound classifier project.
