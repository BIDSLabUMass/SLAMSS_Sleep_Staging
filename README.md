# AI-Driven Sleep Staging from Actigraphy and HeartRate


Tzu-An Song, Samadrita Roy Chowdhury, Masoud Malekzadeh, Stephanie Harrison, Terri Blackwell Hoge, Susan Redline, Katie L. Stone, Richa Saxena, Shaun M. Purcell, Joyita Dutta 

Sleep is an important indicator of a person’s health, and its accurate and cost-effective quantification is of great value in healthcare. The gold standard for sleep assessment and the clinical diagnosis of sleep disorders is polysomnography (PSG). However, PSG requires an overnight clinic visit and trained technicians to score the obtained multimodality data. Wrist-worn consumer devices, such as smartwatches, are a promising alternative to PSG because of their small form factor, continuous monitoring capability, and popularity. Unlike PSG, however, wearables-derived data are noisier and far less information-rich because of the fewer number of modalities and less accurate measurements due to their small form factor. Given these challenges, most consumer devices perform two-stage (i.e., sleep-wake) classification, which is inadequate for deep insights into a person’s sleep health. The challenging multi-class (three, four, or five-class) staging of sleep using data from wrist-worn wearables remains unresolved. The difference in the data quality between consumer-grade wearables and lab-grade clinical equipment is the motivation behind this study. In this paper, we present an artificial intelligence (AI) technique termed sequence-to-sequence LSTM for automated mobile sleep staging (SLAMSS), which can perform three-class (wake, NREM, REM) and four-class (wake, light, deep, REM) sleep classification from activity (i.e., wrist-accelerometry-derived locomotion) and two coarse heart rate measures—both of which can be reliably obtained from a consumer-grade wrist-wearable device. Our method relies on raw time-series datasets and obviates the need for manual feature selection. We validated our model using actigraphy and coarse heart rate data from two independent study populations: the Multi-Ethnic Study of Atherosclerosis (MESA; N = 808) cohort and the Osteoporotic Fractures in Men (MrOS; N = 817) cohort. SLAMSS achieves an overall accuracy of 79%, weighted F1 score of 0.80, 77% sensitivity, and 89% specificity for three-class sleep staging and an overall accuracy of 70-72%, weighted F1 score of 0.72-0.73, 64-66% sensitivity, and 89-90% specificity for four-class sleep staging in the MESA cohort. It yielded an overall accuracy of 77%, weighted F1 score of 0.77, 74% sensitivity, and 88% specificity for three-class sleep staging and an overall accuracy of 68-69%, weighted F1 score of 0.68-0.69, 60-63% sensitivity, and 88-89% specificity for four-class sleep staging in the MrOS cohort. These results were achieved with feature-poor inputs with a low temporal resolution. In addition, we extended our three-class staging model to an unrelated Apple Watch dataset. Importantly, SLAMSS predicts the duration of each sleep stage with high accuracy. This is especially significant for four-class sleep staging, where deep sleep is severely underrepresented. We show that, by appropriately choosing the loss function to address the inherent class imbalance, our method can accurately estimate deep sleep time (SLAMSS/MESA: 0.61±0.69 hours, PSG/MESA ground truth: 0.60±0.60 hours; SLAMSS/MrOS: 0.53±0.66 hours, PSG/MrOS ground truth: 0.55±0.57 hours;). Deep sleep quality and quantity are vital metrics and early indicators for a number of diseases. Our method, which enables accurate deep sleep estimation from wearables-derived data, is therefore promising for a variety of clinical applications requiring long-term deep sleep monitoring.

<!-- 
Tzu-An Song<sup>1</sup>, Samadrita Roy Chowdhury<sup>1</sup>, Fan Yang<sup>1</sup>, Joyita Dutta<sup>1</sup></br>
<sup>1</sup>Department of Electrical and Computer Engineering, University of Massachusetts Lowell, Lowell, MA, 01854 USA and co-affiliated with Massachusetts General Hospital, Boston, MA, 02114.

The intrinsically low spatial resolution of positron emission tomography (PET) leads to image quality degradation and inaccurate image-based quantitation. Recently developed supervised super-resolution (SR) approaches are of great relevance to PET but require paired low- and high-resolution images for training, which are usually unavailable for clinical datasets. In this paper, we present a self-supervised SR (SSSR) technique for PET based on dual generative adversarial networks (GANs), which precludes the need for paired training data, ensuring wider applicability and adoptability. The SSSR network receives as inputs a low-resolution PET image, a high-resolution anatomical magnetic resonance (MR) image, spatial information (axial and radial coordinates), and a high-dimensional feature set extracted from an auxiliary CNN which is separately-trained in a supervised manner using paired simulation datasets. The network is trained using a loss function which includes two adversarial loss terms, a cycle consistency term, and a total variation penalty on the SR image. We validate the SSSR technique using a clinical neuroimaging dataset. We demonstrate that SSSR is promising in terms of image quality, peak signal-to-noise ratio, structural similarity index, contrast-to-noise ratio, and an additional no-reference metric developed specifically for SR image quality assessment. Comparisons with other SSSR variants suggest that its high performance is largely attributable to simulation guidance.

Published in: Neural Networks

Pages: 83 - 91

DOI: 10.1016/j.neunet.2020.01.029

The paper can be found [here](https://www.sciencedirect.com/science/article/abs/pii/S0893608020300393?via%3Dihub).

Our previous work is also available [here](https://github.com/alansoong200/SR_PET_CNN) on github.


## Prerequisites

This code uses:

- Python 2.7
- Pytorch 0.4.0
- matplotlib 2.2.4
- numpy 1.16.4
- scipy 1.2.1
- NVIDIA GPU
- CUDA 8.0
- CuDNN 7.1.2

## Dataset

BrainWeb (Simulated Brain Database):
https://brainweb.bic.mni.mcgill.ca/brainweb/

Alzheimer’s Disease Neuroimaging Initiative (ADNI) (Clinical Database):
http://adni.loni.usc.edu/

## Citation
If this work inspires you, please cite our paper:

	@article{
		song_pet_2020,
		title = {{PET} image super-resolution using generative adversarial networks},
		volume = {125},
		issn = {0893-6080},
		url = {http://www.sciencedirect.com/science/article/pii/S0893608020300393},
		doi = {10.1016/j.neunet.2020.01.029},
		language = {en},
		urldate = {2020-07-02},
		journal = {Neural Networks},
		author = {Song, Tzu-An and Chowdhury, Samadrita Roy and Yang, Fan and Dutta, Joyita},
		month = may,
		year = {2020},
		keywords = {Super-resolution, CNN, GAN, Multimodality imaging, PET, Self-supervised},
		pages = {83--91},
	}
}
-->
## UMASS_Amherst_BIDSLab
Biomedical Imaging & Data Science Laboratory

Lab's website:
http://www.bidslab.org/index.html


Email: bidslab(at)gmail.com,
       tzuansong(at)umass.edu.
