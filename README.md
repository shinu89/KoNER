# KoNER

KoNER (Korean Named Entity Recognizer)는 한국어 개체명 인식기이다. 

## Requirements

* Python 2.7
* Theano
* Numpy
* [KoNLPy](http://konlpy-ko.readthedocs.io/ko/v0.4.3/)
* [MeCab](https://bitbucket.org/eunjeon/mecab-ko/overview) 

## Installation
* Mecab installation
```
>> sudo bash install_mecab.sh
```
* Other libraries (theano, numpy, konlpy)
```
>> pip install -r requirements.txt
```

## Pre-trained Model
```
>> tar -xvf model.tar.gz
```

## Run Command
```
python main.py -i [input_file_path] -o [output_file_path] -m [model_path] -p [input type 0(=raw), 1(=pos)]
```
* Example
```
>> python main.py -i './data/test_input.txt' -o './data/test_result.txt' -m './model' -p 0
```
