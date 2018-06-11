# SpeechRecognition
Project realized by Anna Legierska and Karolina Lebiecka.


Simple speech recognition system can be implemented using DTW and MFCC. System is able to recognize 13 polish words pronuanced by different speakers

Set of words: GARAZ, MUZYKE, NASTROJ, OTWORZ, PODNIES, ROLETY, SWIATLO, TELEWIZOR, WLACZ, WYLACZ, ZAMKNIJ, ZAPAL, ZROB

First install dtw. The latest stable release in available on PyPI by typing:
pip install dtw

DTW - Dynamic Time Wrapping
For classify objects DTW - Dynamic Time Wrapping was used.

DWT algorithm measures similarity between two signals with different signal length.
Classifier measures distance between testing signal and every signal in training set and assigns object to the class of signal, where the least distance was found.

MFCC - Mel-Frequency Cepstral Coefficients
MFCC algoritm is used for feature extraction.

MFCCs are the coefficients which represents short-term power spectrum of a data according to MEL scale of frequency, which is based on human hearing perception.

Sources: https://www.researchgate.net/publication/260762671_Speech_recognition_using_MFCC_and_DTW
