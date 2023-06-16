# Retrosynthetic-Planning
Machine Learning project spring 2023

Task1: Single-step retrosynthesis prediction: 

Firstly,

```shell
python reprocess.py
```

```shell
python data_reprocess.py
```

Then 

```shell
cd single_step_retrosynthesis_prediction
python main.py
```



Task2: Molecule Evaluation

```shell
cd molecule_evaluation
python molecule_evaluation.py
```

You can open ***molecule_evaluation.py*** to check more parameter settings.




Task3: Multi-Step retrosynthesis planning

We use the Retro* code. The project's Github [link](https://github.com/binghong-ml/retro_star).  
Firstly follow the README in the ***retro_star*** folder. Then move the ***Multi-Step task*** folder under the path ***./retro_star/retro_star***. Finally run the api.py to get the result.   
Our result is in the ***./retro_star/retro_star/task3.log***
