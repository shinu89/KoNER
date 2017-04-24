# KoNER

KoNER (Korean Named Entity Recognizer)는 한국어 개체명 인식기이다. 

## Requirements

* Python 2.7
* Theano
* Numpy

### Command
```
python main.py -i [input_file_path] -o [output_file_path] -m [model_path] -p [input type 0(=raw), 1(=pos)]
```
* tar -xvf model.tar.gz
* python main.py -i './data/test_input.txt' -o './data/test_result.txt' -m './model' -p 0
