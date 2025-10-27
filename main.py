import warnings
warnings.filterwarnings("ignore")

from laptopPrice.pipeline.training_pipeline import TrainingPipeline

# Run the training pipeline
training_pipeline_obj = TrainingPipeline()
training_pipeline_obj.run_training_pipeline()