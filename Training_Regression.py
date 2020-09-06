import numpy as np
import torch

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, RunningAverage
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint, Checkpoint


def run_training(model,
                 dataset,
                 t_loader,
                 v_loader,
                 max_epochs,
                 criterion=torch.nn.SmoothL1Loss(),
                 flip=False,
                 flip_prob=0, 
                 initial_lr=0.001,
                 lr_sched=None,
                 optim=None,
                 pbar=True,
                 checkpoint=True,
                 save_as="default",
                 weight_decay=1e-5,
                 ):
    # Set seed for pseudo-random numbers
    seed = 121
    np.random.seed = seed
    torch.manual_seed(seed)

    # Dict for storing epoch metrics
    lists = {}
    lists["Training Loss"] = []
    lists["Validation Loss"] = []
    lists["Learning Rate"] = []
    lists["Training RMSE"] = []
    lists["Validation RMSE"] = []

    # Dicts for storing best epoch validation metrics and model.state_dict()
    low = {}
    low['Val'] = 0
    low['Epoch'] = 0
    low['Model'] = 0

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Define Optimizer, Loss
    criterion = criterion
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=initial_lr,
                                 weight_decay=weight_decay)

    # If resuming from previous run, load optim and lr_sched
    if optim != None:
        optimizer.load_state_dict(optim)

    if lr_sched != None:
        lr_scheduler.load_state_dict(lr_sched)

    # Send model to gpu/cpu if not already done
    model.to(device)

    def process_function(engine, batch):
        # Determine whether dataset images flipped and probability
        dataset.flip = flip
        dataset.flip_prob = flip_prob

        # Main training loop
        model.train()
        optimizer.zero_grad()
        x, y = batch
        x = x.float()
        y = y.float()
        y = y[:, 1, :]
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def eval_function(engine, batch):
        # Reset flip - images should not be flipped for eval
        dataset.flip = False
        dataset.flip_prob = flip_prob

        # Main eval loop
        model.to(device)
        model.eval()
        with torch.no_grad():
            x, y = batch
            x = x.float()
            y = y.float()
            y = y[:, 1, :]
            x = x.to(device)
            y = y.to(device)
            model.to(device)
            y_pred = model(x)
            return y_pred, y

    # define engines
    trainer = Engine(process_function)
    train_evaluator = Engine(eval_function)
    validation_evaluator = Engine(eval_function)

    # Attach loss functions to engines
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    Loss(criterion).attach(train_evaluator, 'bce')
    Loss(criterion).attach(validation_evaluator, 'bce')

    # Additional metrics for classification
    from ignite.metrics import RootMeanSquaredError
    RootMeanSquaredError().attach(train_evaluator, 'rmse')
    RootMeanSquaredError().attach(validation_evaluator, 'rmse')

    # Prints progress bar if pbar=True
    if pbar:
        pbar = ProgressBar(persist=True, bar_format="")
        pbar.attach(trainer, ['loss'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        # Run train evaluator on training dataset and log results
        train_evaluator.run(t_loader)
        metrics = train_evaluator.state.metrics
        avg_bce = metrics['bce']
        rmse = metrics ['rmse']
        lists["Training Loss"].append(avg_bce)
        lists["Training RMSE"].append(rmse)
        lists["Learning Rate"].append(optimizer.param_groups[0]['lr'])
        if pbar:
            pbar.log_message(
                """=====================================================================\
            \nEpoch: {}\nLearning Rate: {}\nTraining -- Avg loss: {:.4f}, RMSE: {:.4f}"""
                .format(engine.state.epoch, optimizer.param_groups[0]['lr'],
                        avg_bce, rmse))
        else:
            print(
                """===================================================================\
        \nEpoch: {}\nTraining -- Avg loss: {:.4f}, RMSE: {:.4f}""".format(
                    engine.state.epoch, avg_bce, rmse))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        # Run validation evaluator on validation dataset and log results
        validation_evaluator.run(v_loader)
        metrics = validation_evaluator.state.metrics
        avg_bce = metrics['bce']
        rmse = metrics ['rmse']
        lists["Validation Loss"].append(avg_bce)
        lists["Validation RMSE"].append(rmse)

        # Log best model and update if better validation loss achieved
        if avg_bce < low['Val'] or engine.state.epoch == 1:
            low['Val'] = avg_bce
            low['Epoch'] = engine.state.epoch
            low['Model'] = model.state_dict()

        if pbar:
            pbar.log_message(
                "Validation -- Avg loss: {:.4f}, RMSE: {:.4f}".format(
                    avg_bce, rmse))
            pbar.n = pbar.last_print_n = 0

        else:
            print("Validation -- Avg loss: {:.4f}, RMSE: {:.4f}".format(
                avg_bce, rmse))

    # Create checkpoint with model, optim and lr_sched
    if checkpoint:
      checkpointer = ModelCheckpoint(
          '/content/drive/My Drive/Dissertation Files/Models',
          'model' + save_as,
          n_saved=1,
          create_dir=True,
          save_as_state_dict=True,
          require_empty=False)
      trainer.add_event_handler(
          Events.EPOCH_COMPLETED, checkpointer, {
              save_as: model,
              save_as + "optim": optimizer,
              save_as + "lr_sched": lr_scheduler
          })
      
    
    # Run training loops until max_epochs reached
    trainer.run(t_loader, max_epochs=max_epochs)

    # Save model with best validation loss
    model_name_best = save_as + '_LowestLossEpoch{}'.format(low['Epoch'])
    torch.save({'model': low['Model']},
               '/content/drive/My Drive/Dissertation Files/Models/' +
               model_name_best)
    
    # Save model at end of max_epochs
    model_name_last = save_as + '_FinalEpoch{}'.format(max_epochs)
    torch.save({'model': model.state_dict()},
               '/content/drive/My Drive/Dissertation Files/Models/' +
               model_name_last)

    print('''\nLowest Validation Loss: {:.4f} at Epoch {}
    \nBest model state_dict saved as /content/drive/My Drive/Dissertation Files/Models/{}
    '''.format(low['Val'], low['Epoch'], model_name_best))

    return lists