# AutoKeras

# https://github.com/yjucho1/articles/blob/master/auto-keras/readme.md

# install : git (git 버전이 keras 개체 변환 가능)
# pip install git+https://github.com/jhfjhfj1/autokeras.git
# 문제점 : torch 0.4.1 버전
# https://github.com/jhfjhfj1/autokeras/issues/41
# pip install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-win_amd64.whl
# pip install torchvision

from keras.datasets import cifar10
#from autokeras.classifier import ImageClassifier
from autokeras.image_supervised import ImageClassifier

# pip버전 에러 : THCudaCheck FAIL / multiprocessing.pool.MaybeEncodingError
# https://github.com/jhfjhfj1/autokeras/issues/76 (CUDA 설정상 오류)
# 경로 : C:\Users\ChopperY\AppData\Local\Programs\Python\Python36\Lib\site-packages\autokeras\search.py
# 교체대상1
# train_results = pool.map_async(train, [(graph, train_data, test_data, self.trainer_args, 
#                                                os.path.join(self.path, str(model_id) + '.png'), self.verbose)])
# 교체결과1
# train_results = train((graph, train_data, test_data, self.trainer_args, os.path.join(self.path, str(model_id) + '.png'), self.verbose))
# 교체대상2
# accuracy, loss, graph = train_results.get()[0]
# 교체결과2 
# accuracy, loss, graph = train_results

# https://github.com/jhfjhfj1/autokeras/issues/76
# if you are on windows, torch with CUDA and multiprocessing do not seem to work well together.
# Also please try to wrap your code in trainalutokeras_raw.py in:
# if __name__ == "__main__"

if __name__ == '__main__':
    # 1. load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # 2. define classifier : 경로를 설정하지 않으면 default경로(/tmp/autokeras)로 저장됨
    # git 버전 default경로 : C:\Users\ChopperY\AppData\Local\Temp\autokeras
    clf = ImageClassifier(verbose=True, searcher_args={'trainer_args':{'max_iter_num':5}})
    #clf = ImageClassifier(verbose=True, path='d:/tmp/autokeras/', searcher_args={'trainer_args':{'max_iter_num':5}})
    
    # 3. fitting
    clf.fit(x_train, y_train, time_limit=24 * 60 * 60)
    
    # 3-1. Load saved model
    '''
    # if you reloaded your saved clf, y_encoder & data_transformer should be defined like following.
    from autokeras.preprocessor import OneHotEncoder, DataTransformer
    from autokeras.constant import Constant
    clf.y_encoder = OneHotEncoder()
    clf.y_encoder.fit(y_train)
    clf.data_transformer = DataTransformer(x_train, augment=Constant.DATA_AUGMENTATION)
    
    #print(clf.get_best_model_id())
    
    searcher = clf.load_searcher()
    #print(searcher.history)
    
    # 3-2. fitting finally and saving model
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=False, trainer_args={'max_iter_num': 10})
    y = clf.evaluate(x_test, y_test)
    print(y)
    
    clf.save_searcher(searcher)
    '''
    
    # 4. 저장된 clf, searcher 다시 불러와 작업 (3, 3-1, 3-2 주석처리 필요)
    '''
    searcher = clf.load_searcher()
    from pprint import pprint
    #pprint(searcher.history)
    
    graph = searcher.load_best_model()
    ## Or you can load graph by id
    # graph = searcher.load_model_by_id(16)
    #pprint(graph)
    
    torch_model = graph.produce_model()
    pprint(torch_model)
    
    keras_model = graph.produce_keras_model()
    keras_model.summary()
    
    # keras 모델 tuning
    import keras.utils as utils
    y_test = utils.to_categorical(y_test, num_classes=None)
    y_train = utils.to_categorical(y_train, num_classes=None)

    keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    keras_model.evaluate(x_test, y_test)

    keras_model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))
    '''
