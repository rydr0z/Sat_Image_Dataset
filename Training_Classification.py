import numpy as np
import torch

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, RunningAverage
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint, Checkpoint


def run_training(
        model,
        dataset,
        t_loader,
        v_loader,
        max_epochs=70,
        flip=False,
        flip_prob=0,
        initial_lr=0.001,
        lr_sched=None,
        optim=None,
        pbar=True,
        checkpoint=True,
        save_as='default',
        weight_decay=1e-5,
        rgb=False):
    '''
    Runs training loops for max_epochs.
    Arguments:
      model: model to train
      dataset: torch Dataset with image and label pairs
      default loader: torch DataLoader that gets training samples without any weighted sampling
      t_loader: either the same as default loader or one that does weighted sampling
      v_loader: torch DataLoader that gets validation samples from dataset
      max_epochs: number of epochs to run training
      flip: whether to augment data by flipping images
      flip_prob: if flipping images, probability of flip
      initial_lr: beginning learning rate and max for cosine annealing lr scheduler
      lr_sched: Optional - to load saved lr_sched
      optim: Optional - to load saved optim
      pbar: prints a progress bar for each epoch
      checkpoint: whether to save model, lr_sched and optim each epoch
      save_as: name of model file to save
      weight_decay: L2 weight decay for optimizer

    Returns:
      lists: dictionary with Training Loss, Validation Loss, Learning Rates, 
            Training Accuracy and Validation Accuracy

    '''

    # Set seed for pseudo-random numbers
    seed = 121
    np.random.seed = seed
    torch.manual_seed(seed)

    # Dict for storing epoch metrics
    lists = {}
    lists['Training Loss'] = []
    lists['Validation Loss'] = []
    lists['Learning Rate'] = []
    lists['Training Accuracy'] = []
    lists['Validation Accuracy'] = []

    # Dicts for storing best epoch validation metrics and model.state_dict()
    low = {}
    low['Val'] = 0
    low['Epoch'] = 0
    low['Model'] = 0
    
    high = {}
    high['Val'] = 0
    high['Accuracy'] = 0
    high['Model'] = 0

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Define Optimizer, Loss and LR Scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=initial_lr,
                                 weight_decay=weight_decay)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #    optimizer, 10, T_mult=2)

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
        if rgb:
            x = x[:,0:3,:,:]
        x = x.float()
        y = y.type(torch.LongTensor)
        y = y[:, 2, :].squeeze()
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        # Next learning rate from lr_scheduler
        #i = engine.state.iteration / len(t_loader)
        #lr_scheduler.step(i)

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
            if rgb:
                x = x[:,0:3,:,:]
            x = x.float()
            y = y.type(torch.LongTensor)
            y = y[:, 2, :].squeeze()
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
    from ignite.metrics import Accuracy, Fbeta
    Accuracy().attach(train_evaluator, 'accuracy')
    Accuracy().attach(validation_evaluator, 'accuracy')
    Fbeta(0.5, True).attach(train_evaluator, 'Fbeta')
    Fbeta(0.5, True).attach(validation_evaluator, 'Fbeta')
    
    # Prints progress bar if pbar=True
    if pbar:
        pbar = ProgressBar(persist=True, bar_format='')
        pbar.attach(trainer, ['loss'])
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        # Run train evaluator on training dataset and log results
        train_evaluator.run(t_loader)
        metrics = train_evaluator.state.metrics
        avg_bce = metrics['bce']
        accuracy = metrics['accuracy']
        fbeta = metrics['Fbeta']
        lists['Training Loss'].append(avg_bce)
        lists['Training Accuracy'].append(accuracy)
        lists['Learning Rate'].append(optimizer.param_groups[0]['lr'])
        if pbar:
            pbar.log_message(
                '''=====================================================================\
            \nEpoch: {} | Learning Rate: {}\nTraining   | Avg loss: {:.4f} | Accuracy: {:.2f}% | Fbeta: {:.4f}'''
                .format(engine.state.epoch, optimizer.param_groups[0]['lr'],
                        avg_bce, accuracy * 100, fbeta))
        else:
            print(
                '''===================================================================\
        \nEpoch: {}\nTraining   | Avg loss: {:.4f} | Accuracy: {:.2f}% | Fbeta: {:.4f}'''
                .format(engine.state.epoch, avg_bce, accuracy * 100, fbeta))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        # Run validation evaluator on validation dataset and log results
        validation_evaluator.run(v_loader)
        metrics = validation_evaluator.state.metrics
        avg_bce = metrics['bce']
        accuracy = metrics['accuracy']
        fbeta = metrics['Fbeta']
        lists['Validation Loss'].append(avg_bce)
        lists['Validation Accuracy'].append(accuracy)

        # Log best model and update if better validation loss achieved
        if avg_bce < low['Val'] or engine.state.epoch == 1:
            low['Val'] = avg_bce
            low['Accuracy'] = accuracy
            low['Epoch'] = engine.state.epoch
            low['Model'] = model.state_dict()
            
        if accuracy > high['Accuracy'] or engine.state.epoch == 1:
            high['Val'] = avg_bce
            high['Accuracy'] = accuracy
            high['Epoch'] = engine.state.epoch
            high['Model'] = model.state_dict()

        if pbar:
            pbar.log_message(
                'Validation | Avg loss: {:.4f} | Accuracy: {:.2f}% | Fbeta: {:.4f}'
                .format(avg_bce, accuracy * 100, fbeta))
            pbar.n = pbar.last_print_n = 0
        else:
            print(
                'Validation | Avg loss: {:.4f} | Accuracy: {:.2f}% | Fbeta: {:.4f}'
                .format(avg_bce, accuracy * 100, fbeta))

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
                save_as + 'optim': optimizer,
                save_as + 'lr_sched': lr_scheduler
            })

    # Run training loops until max_epochs reached
    trainer.run(t_loader, max_epochs=max_epochs)

    # Save model with best validation loss
    model_name_best = save_as + '_LowestLossEpoch{}'.format(low['Epoch'])
    torch.save({'model': low['Model']},
               '/content/drive/My Drive/Dissertation Files/Models/' +
               model_name_best)
    
    # Save model with best validation accuracy
    model_name_best2 = save_as + '_HighestAccuracyEpoch{}'.format(low['Epoch'])
    torch.save({'model': high['Model']},
               '/content/drive/My Drive/Dissertation Files/Models/' +
               model_name_best2)

    # Save model at end of max_epochs
    model_name_last = save_as + '_FinalEpoch{}'.format(max_epochs)
    torch.save({'model': model.state_dict()},
               '/content/drive/My Drive/Dissertation Files/Models/' +
               model_name_last)

    print('''\nLowest Validation Loss: {:.4f} at Epoch {} with {:.2f}% Accuracy
        \nBest model state_dict saved as /content/drive/My Drive/Dissertation Files/Models/{}
        '''.format(low['Val'], low['Epoch'], low['Accuracy']*100, model_name_best))
    print('''\nHighest Validation Accuracy: {:.2f}% at Epoch {} with {:.4f} Loss
        \nBest model state_dict saved as /content/drive/My Drive/Dissertation Files/Models/{}
        '''.format(high['Accuracy']*100, high['Epoch'], high['Val'], model_name_best2))

    return lists

def test_classification(model, dataset, num_images, device, rgb=False):
    from sklearn.metrics import classification_report, accuracy_score, \
        balanced_accuracy_score, f1_score, plot_confusion_matrix
    pred_actual_list = []
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=False)
    iter_loader = iter(loader)
    model.to(device)
    for i in range(0, num_images):
        images, labels = iter_loader.next()
        if rgb:
            images = images[:,0:3,:,:]
        images = images.float()
        images = images.to(device)
        labels = labels.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            pred_val1, pred_lab1 = torch.max(outputs, dim=1)
            pred_val2, pred_lab2 = torch.max(outputs[outputs != pred_val1], dim=0)
            outputs1 = outputs[outputs != pred_val1]
            pred_val3, pred_lab3 = torch.max(
                outputs1[outputs1 != pred_val2], dim=0)
            pred = int(pred_lab1.item())
            pred2 = int(pred_lab2.item())
            pred3 = int(pred_lab3.item())
        pop = labels[0][2].item()
        index = labels[0][0].item()
        pred_actual_list.append([index, pred, pop, pred2, pred3])

    indices = [item[0] for item in pred_actual_list]
    preds = [item[1] for item in pred_actual_list]
    pop = [item[2] for item in pred_actual_list]
    preds2 = [item[3] for item in pred_actual_list]
    preds3 = [item[4] for item in pred_actual_list]

    pop_np = np.array(pop)
    preds_np = np.array(preds)
    preds2_np = np.array(preds2)
    preds3_np = np.array(preds3)

    #
    accuracy = ((preds_np == pop_np).sum() / preds_np.size) * 100
    accuracy2 = (np.logical_or((preds_np == pop_np),
                               (preds2_np == pop_np))).sum() / preds2_np.size * 100
    accuracy3 = (np.logical_or(np.logical_or((preds_np == pop_np),
                                             (preds2_np == pop_np)), (preds3_np == pop_np))).sum() / preds3_np.size * 100

    cl_report = classification_report(pop, preds)
    accuracy_sc = accuracy_score(pop, preds) * 100
    bal_accuracy = balanced_accuracy_score(pop, preds) * 100
    f1_score = f1_score(pop, preds, average='weighted')

    print('''Overall Accuracy (Top 1, Top 2, Top 3): {:2f}%, {:2f}%, {:2f}%
    \nAccuracy: {:.2f}%
    \nBalanced Accuracy: {:.2f}%
    \nF1 Score: {:.2f}
    \n
    \n{}'''.format(accuracy, accuracy2, accuracy3,accuracy_sc, bal_accuracy, f1_score,  cl_report)
    )
    return pred_actual_list