# AutoKeras 예제입니다.
# 작성일자 : 2018. 9. 29
# 예제 작성간 발생한 문제점과 해결책들을 묶어 한번에 사용하기 편하게 만들고자 했습니다.
# 참조, 수정내용 공유는 언제나 환영합니다. 단, 출처는 반드시 표기해 주세요.
# 메인 참조자료 : https://github.com/yjucho1/articles/blob/master/auto-keras/readme.md (감사합니다)

# 환경 : Windows10 64bit 영문버전, python 3.6, TensorFlow-gpu 세팅

# Install : pip 버전과 git 버전 가능 (git 버전이 keras 개체로 변환이 가능하므로 권장합니다)
# git버전 인스톨 : pip install git+https://github.com/jhfjhfj1/autokeras.git

# Install시 문제점 : torch 0.4.1 버전이 맞지 않아 생기는 문제
# 참조 페이지 : https://github.com/jhfjhfj1/autokeras/issues/41
# 해결방안 : torch 버전을 맞춰 install해준다.
# pip install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-win_amd64.whl
# pip install torchvision

from keras.datasets import cifar10
# pip 버전에서 classifier 참조 위치
#from autokeras.classifier import ImageClassifier
# git 버전에서 classifier 참조 위치
from autokeras.image_supervised import ImageClassifier

# pip 버전 에러 : THCudaCheck FAIL / multiprocessing.pool.MaybeEncodingError
# 문제점 : CUDA 설정상 오류로 추정됩니다.
# 참조 페이지 : https://github.com/jhfjhfj1/autokeras/issues/76
# 해결책 : \autokeras\search.py 파일의 특정 부분을 수정
# 교체대상1
# train_results = pool.map_async(train, [(graph, train_data, test_data, self.trainer_args, 
#                                                os.path.join(self.path, str(model_id) + '.png'), self.verbose)])
# 교체결과1
# train_results = train((graph, train_data, test_data, self.trainer_args, os.path.join(self.path, str(model_id) + '.png'), self.verbose))
# 교체대상2
# accuracy, loss, graph = train_results.get()[0]
# 교체결과2 
# accuracy, loss, graph = train_results

# 코드 추가 필요 : Windows를 사용하는 경우
# 참조 페이지 : https://github.com/jhfjhfj1/autokeras/issues/76
# if you are on windows, torch with CUDA and multiprocessing do not seem to work well together.
# Also please try to wrap your code in trainalutokeras_raw.py in:
# if __name__ == "__main__"

if __name__ == '__main__':
    # 1. Load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # 2. Define classifier : 경로를 설정하지 않으면 default 경로로 저장됨
    # pip 버전 default 경로 : (AutoKeras를 실행한 위치 root)\tmp\autokeras
    # git 버전 default경로 : (AutoKeras를 설치한 위치 username)\AppData\Local\Temp\autokeras
    clf = ImageClassifier(verbose=True, searcher_args={'trainer_args':{'max_iter_num':5}})
    #clf = ImageClassifier(verbose=True, path='d:/tmp/autokeras/', searcher_args={'trainer_args':{'max_iter_num':5}})
    
    # 3. Fitting
    # time_limit : 초단위, 시간이 지나면 작동을 자동으로 멈춥니다.
    clf.fit(x_train, y_train, time_limit=24 * 60 * 60)
    
    # 3-1. Load saved model (3번 항목 실행후 3 주석처리 필요)
    
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
    
    
    # 4. 저장된 clf, searcher 다시 불러와 작업 (3번 항목 실행후 3, 3-1, 3-2 주석처리 필요)
    
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
