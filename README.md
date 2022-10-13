소 키포인트
Animal Datathon Korea 2021

정확두 부문 3위

기존에 많은 keypoint 대회에서 HRNET이 상위권인것을 보고 HRNET기준으로 진행 결정
동물 기준으로 keypoint 학습이 된 pretrain weight도 찾았으나 성능은 향상되지 않았기 때문에 사람기준 pretrain된 weight 기준으로
여러가지 augmentation을 적용해보고 가장 좋은 augmentation 조합기준으로 진행함( random noise , random contrast, random brightness, paste_cutout)

막바지에 아쉬운점은 test-time augmentation 적용시에 여러 output 가운데 validation 기준으로 봤을 때 정답에 가까운 output도 있고 오히려 정답에서 멀어진 output도 있는데
단순 결과물의 average를 사용하는 것이 아니라 소의 keypoint 분포에 맞는 최적의 output을 선택할 수 있도록 딥러닝 OR 머신러닝 기반 기법을 하나 더 사용하여 최적의 output을 선택하도록 했더라면 더 우수한 결과를 낼 수 있었을거 같다는 생각이 들어서 아쉬움이 남음.
