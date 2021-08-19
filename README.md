# bp-estimation-unet
 
## Objective
This project aims to use Electrocardiogram data (ECG) and Photoplethysmography data (PPG) to provide a continuous prediction of arterial blood pressure (ABP). The current methods for measuring blood pressure are invasive, including the use of an arterial line or an inflatable cuff. Deriving the blood pressure signal from the synthesis of other biosignals would allow for continuous blood pressure monitoring in a non-invasive manner.

## Dataset
The dataset used was obtained from UCI Machine Learning Repository (<a href="https://archive.ics.uci.edu/ml/datasets/Cuff-Less+Blood+Pressure+Estimation">Link</a>). The raw electrocardiogram (ECG), photoplethysmograph (PPG), and arterial blood pressure (ABP) signals are originally collected from the physionet.org MIMIC II Dataset (<a href="https://archive.physionet.org/physiobank/database/mimic2wdb/matched/">Link</a>). 

## Model
The model being used is a 1-D Fully Convolutional Neural Network (FCN) which takes in a sequence and returns a sequence. Here we use a 1-D version of the MultiResUNet (<a href="https://github.com/nibtehaz/MultiResUNet">Link</a>). Performing convolutions along the waveform will add the capacity for the model to learn the shape and structure of the input signals. 

The model was trained by dividing up the waveforms into segments containing two heartbeats (peak-peak). One model was created using the PPG sequences as the input and the ABP sequences as the output. Another model was created using both PPG and ECG sequences as the output. The performance of the model is similar to an autoencoder, where the inputs are processed down into a latent representation, which is then used to re-create the target ABP sequence. The loss is computed by comparing the recreated ABP waveform to the ground truth ABP waveform.

## Further Steps

- Look into using more data (from MIMIC-II)
- Properly separating data by each patient to be able to validate the model on test patients
- Look into LSTM Networks
- Try training the encoder as an autoencoder using PPG -> PPG. Then freeze the encoder layers and train only the decoder layer on PPG -> ABP.

## Demo

https://user-images.githubusercontent.com/72168799/130130348-b980843a-4e38-4bdd-89d5-1e2a5486cc68.mp4
