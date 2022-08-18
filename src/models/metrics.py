import segmentation_models_pytorch as smp


METRICS = {
    'iou_score': {
        'func': smp.metrics.iou_score,
        'reduction': 'micro'
    },
    'f1_score': {
        'func': smp.metrics.f1_score,
        'reduction': 'micro'
    },
    'accuracy': {
        'func': smp.metrics.accuracy,
        'reduction': 'micro'
    },
    'recall': {
        'func': smp.metrics.recall,
        'reduction': 'micro-imagewise'
    },
}
