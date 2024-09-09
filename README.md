# BemAgro_IA_challenge
Este repositório é dedicado à solução do desafio técnico de IA, parte do processo seletivo para vaga de IA na BemAgro.

A descrição do desafio está em Desafio - IA.pdf

Após quebrar o ortomosaico em imagens menores, usei o índice ExG combinado com o método de Otsu para binarizar as imagens e fazer as máscaras de segmentação. Essa é uma técnica comum que eu já conhecia de artigos em minha experiência de pesquisa.

Para o modelo, utilizei o repositório Segmentation Models (https://github.com/qubvel/segmentation_models), que possui vários modelos de segmentação para serem facilmente utilizados, assim como métricas e funções loss mais adequadas para segmentação. Nesse caso, decidi utilizar o Intersection over Union (IoU), métrica bastante utilizada para esse tipo de tarefa. O modelo escolhido foi a combinação MobilenetV2-Unet, pois é um modelo de segmentação com um bom equilíbrio entre precisão e desempenho em tempo real.

Para melhorar a generalização do modelo, implementei algumas operações de data augmentation durante o treinamento. Esta técnica de augmentation on-the-fly ajuda não só na generalização, mas também na prevenção de overfitting do modelo sem a necessidade de uma quantidade muito grande de dados.

Após 100 épocas, o modelo alcançou um IoU de 90,55% nos dados de validação.

O modelo poderia ter uma generalização melhor caso as operações de augmentation fossem escolhidas com maior cuidado. Além disso, poderia ser testado outros modelos mais pesados, já que esse tipo de tarefa geralmente não é feito em tempo real, permitindo modelos maiores.