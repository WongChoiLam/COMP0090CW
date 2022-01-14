import torch

def Evaluation_mask(model, training_data, targets):

    # Get prediction
    prediction = model(training_data)
    masks_hat = torch.argmax(prediction['out'],dim=1)

    # Get Stats
    accuracy = torch.sum(masks_hat == targets)/((torch.sum(masks_hat == targets))+torch.sum(masks_hat != targets))
    precision = torch.sum(masks_hat[targets == 1] == 1)/torch.sum(targets == 1)
    recall = torch.sum(masks_hat[targets == 1] == 1)/(torch.sum(masks_hat[targets == 1] == 1) + torch.sum(masks_hat[targets == 0] == 1))
    F_1 = 2*precision*recall/(precision+recall)

    # IOU
    intersection = torch.logical_and(targets, masks_hat)
    union = torch.logical_or(targets, masks_hat)
    IOU = torch.sum(intersection) / torch.sum(union)

    return precision, recall, accuracy, F_1, IOU