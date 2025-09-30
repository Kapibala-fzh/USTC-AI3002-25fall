import argparse
import numpy as np
from data_utils import DataSet, DataLoader
from submission import (
    load_and_preprocess_data, 
    LinearRegressionModel, 
    LogisticRegressionModel, 
    LinearRegressionTrainer, 
    LogisticRegressionTrainer, 
    linear_regression_analytic
)
from sklearn.metrics import mean_absolute_error, r2_score, f1_score, roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)

    # Mode specification, regression mens linear regression, classification means logistic regression
    parser.add_argument("--mode", type=str, choices=["regression", "classification"], default="regression")

    # Data path and save path
    parser.add_argument("--train_data_path", type=str, default="data/train.csv")
    parser.add_argument("--eval_data_path", type=str, default="data/valid.csv")
    parser.add_argument("--save_dir", type=str, default="saves")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_strategy", type=str, default="step", choices=["epoch", "step"])
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--eval_metric", type=str, default="mae", choices=["mae", "mse", "r2", "acc", "f1", "auc"])

    args = parser.parse_args()
    return args

def evaluate_regression(trainer: LinearRegressionTrainer):
    if not trainer.eval_dataloader:
        return 0., 0.

    total_grd = []
    total_pred = []
        
    for batch_x, batch_y in trainer.eval_dataloader:
        batch_y = batch_y.reshape(-1, 1)
        preds = trainer.model.forward(batch_x).reshape(-1, 1)
        total_grd.append(batch_y)
        total_pred.append(preds)
        
    total_pred = np.concatenate(total_pred, axis=0)
    total_grd = np.concatenate(total_grd, axis=0)

    mae = mean_absolute_error(total_grd, total_pred)
    r2 = r2_score(total_grd, total_pred)

    print(f"Evaluation results on your eval set: mae: {mae:.2f}, R2: {r2:.2f}")
    return mae, r2

def evaluate_classification(trainer: LinearRegressionTrainer):
    if not trainer.eval_dataloader:
        return 0., 0.

    total_grd = []
    total_pred = []
        
    for batch_x, batch_y in trainer.eval_dataloader:
        batch_y = batch_y.reshape(-1, 1)
        preds = trainer.model.forward(batch_x).reshape(-1, 1)
        total_grd.append(batch_y)
        total_pred.append(preds)
        
    total_pred = np.concatenate(total_pred, axis=0)
    total_pred = (total_pred > 0.5).astype(int)
    total_grd = np.concatenate(total_grd, axis=0)

    f1 = f1_score(total_grd, total_pred)
    auc = roc_auc_score(total_grd, total_pred)

    print(f"Evaluation results on your eval set: F1: {f1:.2f}, AUC: {auc:.2f}")
    return f1, auc

def main():
    args = parse_args()
    np.random.seed(args.seed)
    train_features, train_targets = load_and_preprocess_data(args.train_data_path)
    eval_features, eval_targets = load_and_preprocess_data(args.eval_data_path)
    in_features = train_features.shape[1]
    trainset = DataSet(train_features, train_targets, args.mode == "classification")
    evalset = DataSet(eval_features, eval_targets, args.mode == "classification")
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(evalset, batch_size=args.batch_size, shuffle=False)

    if args.mode == "regression":
        model = LinearRegressionModel(in_features, 1)
        trainer = LinearRegressionTrainer(
            model=model, 
            train_dataloader=train_loader, 
            eval_dataloader=eval_loader, 
            save_dir=args.save_dir, 
            learning_rate=args.lr, 
            eval_strategy=args.eval_strategy, 
            eval_steps=args.eval_steps, 
            num_epochs=args.num_epochs, 
            eval_metric=args.eval_metric
        )
        trainer.train()
        evaluate_regression(trainer)
        
        print("*" * 10 + " The results using analytic solution on eval set " + "*" * 10)
        ana_weight, ana_bias = linear_regression_analytic(evalset.features, evalset.targets)
        trainer.model.weight = ana_weight
        trainer.model.bias = ana_bias
        evaluate_regression(trainer)
    else:
        model = LogisticRegressionModel(in_features, 1)
        trainer = LogisticRegressionTrainer(
           model=model, 
            train_dataloader=train_loader, 
            eval_dataloader=eval_loader, 
            save_dir=args.save_dir, 
            learning_rate=args.lr, 
            eval_strategy=args.eval_strategy, 
            eval_steps=args.eval_steps, 
            num_epochs=args.num_epochs, 
            eval_metric=args.eval_metric
        )
        trainer.train()
        evaluate_classification(trainer)

if __name__ == "__main__":
    main()