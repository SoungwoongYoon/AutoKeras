# AutoKeras 예제입니다.

작성일자 : 2018. 9. 29

- 처음 접하시거나 어려운 부분을 공유하고 해결하고자 만들어 보았습니다.

- 예제 작성간 발생한 문제점과 해결책들을 묶어 한번에 사용하기 편하게 만들고자 했습니다.

- 참조, 수정내용 공유는 언제나 환영합니다. 단, 출처는 반드시 표기해 주세요.

AutoKeras 공식 sample : https://autokeras.com/start/

메인 참조자료 : https://github.com/yjucho1/articles/blob/master/auto-keras/readme.md (대단히 감사합니다)

### 환경 : Windows10 64bit 영문버전, python 3.6, TensorFlow-gpu 세팅

### 참조할 사항

Default setting의 특징
- pip 버전은 에러가 보이고 keras 개체로 변환이 안됩니다. 가급적 git 버전으로 설치하기를 권장합니다.
- Epoch : 200회. 단, 최종 5회 실행에서 loss decrese가 관찰되지 않으면 자동으로 다음 model로 실행됩니다.
- 현재로서는 CNN 기반의 Image Classifier만 있는 것 같습니다. (물론 외부 wrapping으로 다른 Deep learning도 가능할 듯)

Hyperparameter tuning을 자동화하는 부분은 높이 살만 합니다.
- Google AutoML를 써보지 못한 상황에서의 판단입니다.
- 특히 Image Classification에서 필요한 각 layer들을 고려하여 구성하는 것으로 보입니다.

실행하는데 시간이 상당히 소요됩니다. (24시간 실행, 정밀도 기준 평가)
- 현재 다양한 방법으로 실행 테스트를 진행하고 있습니다.
- 방법 1 : Epoch 횟수를 제한하고 (예: 5회) 많은 model을 만들어보는 방법
- 방법 2 : Default setting을 이용, 각 model의 정밀도를 보는 방법
