# 742
742 Group Project


To run you will need to install the openai libraries. pip3 install openai should be all the dependencies needed.

## Fine-tuning Contriever

1. Read through the `src/options.py` file to comprehend what values will be used by the fine-tuning script.
2. Run ```python finetuning.py [options]``` with any option values.
   - Example: ```python --model_path=facebook/contriever --train_data=Data/RedditData --eval_data=Data/RedditData --per_gpu_batch_size=16```
3. The training will commence and store a model in the designated location if options are set correctly.

## Evaluation

1. Generate embeddings for the entire dataset using `gen_embeddings.py` with the path to the saved model, dataset, and output director.
2. Run evaluation script `eval.py` on the embeddings dataset generated in the previous step
   - There are various evaluation functions and options that can be selected at the end of the file.